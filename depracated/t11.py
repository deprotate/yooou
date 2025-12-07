import pandas as pd
import numpy as np
import gc
import warnings
import sys
import os
import joblib
import random
from pathlib import Path
from tqdm import tqdm
from catboost import CatBoostRanker, Pool
import scipy.sparse as sparse
from sklearn.decomposition import TruncatedSVD

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

warnings.filterwarnings('ignore')


class Config:
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = Path("output/models")
    SUBMISSION_DIR = Path("output/submissions")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    SEEDS = [42, 1337, 777, 2024, 100]

    # 50% случайных, 50% популярных (Hard Negatives)
    NEGATIVES_PER_USER = 15

    USE_BERT = True
    BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
    BERT_BATCH_SIZE = 8
    BERT_MAX_LEN = 128

    VAL_SIZE_RATIO = 0.2

    CB_PARAMS = {
        'loss_function': 'YetiRank',
        'iterations': 2500,
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'verbose': 0,
        'eval_metric': 'NDCG:top=20',
        'early_stopping_rounds': 50
    }


class Constants:
    TRAIN_FILENAME = "train.csv"
    TARGETS_FILENAME = "targets.csv"
    CANDIDATES_FILENAME = "candidates.csv"
    USER_DATA_FILENAME = "users.csv"
    BOOK_DATA_FILENAME = "books.csv"
    BOOK_GENRES_FILENAME = "book_genres.csv"
    GENRES_FILENAME = "genres.csv"
    BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"

    COL_USER_ID = "user_id"
    COL_BOOK_ID = "book_id"
    COL_TIMESTAMP = "timestamp"
    COL_HAS_READ = "has_read"
    COL_RELEVANCE = "relevance"
    COL_DESCRIPTION = "description"
    COL_BOOK_ID_LIST = "book_id_list"

    F_SVD_SCORE = "svd_score"
    F_BERT_SIM = "bert_cosine_sim"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if BERT_AVAILABLE:
        torch.manual_seed(seed)


# --- BERT ---
class TextDataset(Dataset):
    def __init__(self, texts, ids, tokenizer, max_len):
        self.texts = texts
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'book_id': torch.tensor(self.ids[item], dtype=torch.long)
        }


def compute_bert_embeddings(desc_df):
    if not BERT_AVAILABLE or not Config.USE_BERT: return {}
    cache_path = Config.PROCESSED_DATA_DIR / "bert_embeddings_full.pkl"
    if cache_path.exists():
        print("Loading cached BERT embeddings...")
        return joblib.load(cache_path)

    # Если кэша нет - этот код выполнится
    print("Computing BERT embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME).to(device)
    model.eval()

    unique_books = desc_df[[Constants.COL_BOOK_ID, Constants.COL_DESCRIPTION]].drop_duplicates(
        subset=Constants.COL_BOOK_ID)
    unique_books[Constants.COL_DESCRIPTION] = unique_books[Constants.COL_DESCRIPTION].fillna("").astype(str)

    dataset = TextDataset(unique_books[Constants.COL_DESCRIPTION].values, unique_books[Constants.COL_BOOK_ID].values,
                          tokenizer, Config.BERT_MAX_LEN)
    loader = DataLoader(dataset, batch_size=Config.BERT_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    embeddings = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            b_ids = batch['book_id'].numpy()
            with torch.cuda.amp.autocast():
                out = model(input_ids=input_ids, attention_mask=mask)
                token_emb = out.last_hidden_state
                input_mask_expanded = mask.unsqueeze(-1).expand(token_emb.size()).float()
                sum_emb = torch.sum(token_emb * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_emb = (sum_emb / sum_mask).float().cpu().numpy()
            for bid, emb in zip(b_ids, mean_emb):
                embeddings[bid] = emb

    joblib.dump(embeddings, cache_path)
    return embeddings


# --- SVD ---
def train_svd_model(train_df):
    print("Training SVD...")
    train_df['weight'] = train_df[Constants.COL_HAS_READ].map({1: 2, 0: 1})
    users = train_df[Constants.COL_USER_ID].unique()
    books = train_df[Constants.COL_BOOK_ID].unique()
    user_map = {u: i for i, u in enumerate(users)}
    book_map = {b: i for i, b in enumerate(books)}
    row = train_df[Constants.COL_USER_ID].map(user_map).values
    col = train_df[Constants.COL_BOOK_ID].map(book_map).values
    data = train_df['weight'].values
    sparse_matrix = sparse.csr_matrix((data, (row, col)), shape=(len(users), len(books)))
    svd = TruncatedSVD(n_components=64, random_state=42)
    u_fac = svd.fit_transform(sparse_matrix)
    i_fac = svd.components_.T
    return u_fac, i_fac, user_map, book_map


def get_svd_score(user_ids, book_ids, u_fac, i_fac, u_map, b_map):
    u_indices = np.array([u_map.get(u, -1) for u in user_ids])
    b_indices = np.array([b_map.get(b, -1) for b in book_ids])
    scores = np.zeros(len(user_ids), dtype=np.float32)
    mask = (u_indices != -1) & (b_indices != -1)
    if mask.sum() > 0:
        scores[mask] = np.sum(u_fac[u_indices[mask]] * i_fac[b_indices[mask]], axis=1)
    return scores


# --- FEATURES ---
def add_audience_features(df, train_history_df, user_meta):
    # (Функция из v6 с чисткой возрастов)
    history_with_meta = train_history_df.merge(user_meta, on=Constants.COL_USER_ID, how='left')
    book_audience = history_with_meta.groupby(Constants.COL_BOOK_ID).agg(
        book_audience_age_mean=('age', 'mean'),
        book_audience_age_std=('age', 'std')
    ).reset_index()

    df = df.merge(book_audience, on=Constants.COL_BOOK_ID, how='left')

    clean_global_median = user_meta['age'].median()
    df['book_audience_age_mean'] = df['book_audience_age_mean'].fillna(clean_global_median)
    df['book_audience_age_std'] = df['book_audience_age_std'].fillna(10.0)

    if 'age' not in df.columns:
        df = df.merge(user_meta[[Constants.COL_USER_ID, 'age']], on=Constants.COL_USER_ID, how='left')

    df['age_filled'] = df['age'].fillna(clean_global_median)
    df['age_diff_with_audience'] = abs(df['age_filled'] - df['book_audience_age_mean'])
    df = df.drop(columns=['age_filled'])
    return df


def add_heuristic_features(df, train_history_df, book_meta):
    print("Generating heuristic features (Authors & Publishers)...")

    # 1. Global Book Stats
    book_stats = train_history_df.groupby(Constants.COL_BOOK_ID).agg(
        book_pop_count=(Constants.COL_USER_ID, 'count'),
        book_global_mean=(Constants.COL_HAS_READ, 'mean')
    ).reset_index()

    # Merge meta to history for calculations
    train_with_meta = train_history_df.merge(book_meta[[Constants.COL_BOOK_ID, 'author_id', 'publisher']],
                                             on=Constants.COL_BOOK_ID, how='left')

    # 2. User-Author Affinity
    user_author_stats = train_with_meta.groupby([Constants.COL_USER_ID, 'author_id'])[Constants.COL_HAS_READ].agg(
        user_author_count='count',
        user_author_mean='mean'
    ).reset_index()

    # 3. User-Publisher Affinity (NEW!)
    # Издательство часто определяет жанр/качество
    user_pub_stats = train_with_meta.groupby([Constants.COL_USER_ID, 'publisher'])[Constants.COL_HAS_READ].agg(
        user_pub_count='count',
        user_pub_mean='mean'
    ).reset_index()

    # --- MERGING ---
    df = df.merge(book_stats, on=Constants.COL_BOOK_ID, how='left')

    # Ensure meta cols exist in df
    cols_needed = ['author_id', 'publisher']
    cols_to_add = [c for c in cols_needed if c not in df.columns]
    if cols_to_add:
        df = df.merge(book_meta[[Constants.COL_BOOK_ID] + cols_to_add], on=Constants.COL_BOOK_ID, how='left')

    df = df.merge(user_author_stats, on=[Constants.COL_USER_ID, 'author_id'], how='left')
    df = df.merge(user_pub_stats, on=[Constants.COL_USER_ID, 'publisher'], how='left')

    # Fill NA
    df['book_pop_count'] = df['book_pop_count'].fillna(0)
    df['book_global_mean'] = df['book_global_mean'].fillna(train_history_df[Constants.COL_HAS_READ].mean())

    df['user_author_count'] = df['user_author_count'].fillna(0)
    df['user_author_mean'] = df['user_author_mean'].fillna(0)

    df['user_pub_count'] = df['user_pub_count'].fillna(0)
    df['user_pub_mean'] = df['user_pub_mean'].fillna(0)

    return df


def calculate_user_bert_profiles(train_history_df, bert_embs):
    if not bert_embs: return {}, 0
    sample_key = next(iter(bert_embs))
    dim = len(bert_embs[sample_key])

    # Оптимизация: не делать DataFrame, если много данных
    # Но для 64GB RAM и 300к строк это быстро
    bert_data = []
    for bid, vec in bert_embs.items():
        bert_data.append([bid] + list(vec))
    bert_df = pd.DataFrame(bert_data, columns=[Constants.COL_BOOK_ID] + [f"b{i}" for i in range(dim)])

    merged = train_history_df.merge(bert_df, on=Constants.COL_BOOK_ID, how='inner')
    user_profiles = merged.groupby(Constants.COL_USER_ID)[[f"b{i}" for i in range(dim)]].mean()

    return {uid: row.values for uid, row in user_profiles.iterrows()}, dim


def build_features(df, u_meta, b_meta, desc_df, svd_data, bert_embs, train_history_full, user_profiles=None):
    print("Building features...")
    df = df.merge(u_meta, on=Constants.COL_USER_ID, how='left')
    cols_to_use = [c for c in b_meta.columns if c not in df.columns or c == Constants.COL_BOOK_ID]
    df = df.merge(b_meta[cols_to_use], on=Constants.COL_BOOK_ID, how='left')

    u_fac, i_fac, u_map, b_map = svd_data
    df[Constants.F_SVD_SCORE] = get_svd_score(df[Constants.COL_USER_ID], df[Constants.COL_BOOK_ID], u_fac, i_fac, u_map,
                                              b_map)

    df = add_heuristic_features(df, train_history_full, b_meta)
    df = add_audience_features(df, train_history_full, u_meta)

    # BERT Similarity
    if user_profiles and bert_embs:
        sample_key = next(iter(bert_embs))
        dim = len(bert_embs[sample_key])

        # Mapping
        u_ids = df[Constants.COL_USER_ID].values
        b_ids = df[Constants.COL_BOOK_ID].values

        def get_u(uid): return user_profiles.get(uid, np.zeros(dim))

        def get_b(bid): return bert_embs.get(bid, np.zeros(dim))

        u_vecs = np.array([get_u(u) for u in u_ids])
        b_vecs = np.array([get_b(b) for b in b_ids])

        dot = np.sum(u_vecs * b_vecs, axis=1)
        n_u = np.linalg.norm(u_vecs, axis=1)
        n_b = np.linalg.norm(b_vecs, axis=1)
        df[Constants.F_BERT_SIM] = dot / (n_u * n_b + 1e-9)

    # BERT Raw (64 dims)
    if bert_embs:
        dim_raw = 64
        emb_matrix = np.zeros((len(df), dim_raw), dtype=np.float32)
        b_ids = df[Constants.COL_BOOK_ID].values
        for i, bid in enumerate(b_ids):
            if bid in bert_embs:
                emb_matrix[i] = bert_embs[bid][:dim_raw]
        bert_df = pd.DataFrame(emb_matrix, columns=[f"bert_{i}" for i in range(dim_raw)], index=df.index)
        df = pd.concat([df, bert_df], axis=1)

    # Clean
    cat_cols = ['gender', 'author_id', 'publisher', 'language']
    for c in cat_cols:
        if c in df.columns: df[c] = df[c].fillna("unk").astype(str)

    num_cols = ['age', 'publication_year', 'avg_rating', 'book_pop_count',
                'user_author_count', 'user_pub_count', 'book_audience_age_mean']
    for c in num_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    return df, cat_cols


# --- PREP HELPERS ---
def load_and_prep():
    print("Loading data...")
    dtype_spec = {Constants.COL_USER_ID: "int32", Constants.COL_BOOK_ID: "int32", Constants.COL_HAS_READ: "int32"}
    train = pd.read_csv(Config.RAW_DATA_DIR / Constants.TRAIN_FILENAME, dtype=dtype_spec,
                        parse_dates=[Constants.COL_TIMESTAMP])
    train[Constants.COL_RELEVANCE] = train[Constants.COL_HAS_READ].map({1: 2, 0: 1}).astype("int8")
    candidates = pd.read_csv(Config.RAW_DATA_DIR / Constants.CANDIDATES_FILENAME,
                             dtype={Constants.COL_USER_ID: "int32"})

    # Clean User Meta
    user_meta = pd.read_csv(Config.RAW_DATA_DIR / Constants.USER_DATA_FILENAME)
    if 'age' in user_meta.columns:
        user_meta.loc[(user_meta['age'] <= 5) | (user_meta['age'] >= 95), 'age'] = np.nan

    book_meta = pd.read_csv(Config.RAW_DATA_DIR / Constants.BOOK_DATA_FILENAME).drop_duplicates(Constants.COL_BOOK_ID)
    book_desc = pd.read_csv(Config.RAW_DATA_DIR / Constants.BOOK_DESCRIPTIONS_FILENAME)
    return train, candidates, user_meta, book_meta, book_desc


def generate_negatives(train_df, all_books, popularity_map=None):
    """
    Генерируем 50% случайных и 50% "тяжелых" (популярных) негативов.
    """
    print("Generating mixed negatives (Random + Hard)...")

    # 1. Найдем Топ-500 популярных книг
    if popularity_map is None:
        pop_counts = train_df[Constants.COL_BOOK_ID].value_counts()
        top_books = pop_counts.head(500).index.values
    else:
        top_books = popularity_map

    user_inter = train_df.groupby(Constants.COL_USER_ID)[Constants.COL_BOOK_ID].apply(set).to_dict()
    all_books_arr = np.array(all_books)

    rows = []
    # Кол-во популярных и случайных
    n_hard = Config.NEGATIVES_PER_USER // 2
    n_random = Config.NEGATIVES_PER_USER - n_hard

    for uid, books in tqdm(user_inter.items()):
        # Random Negatives
        rnd_cands = np.random.choice(all_books_arr, size=n_random + 3)
        cnt = 0
        for b in rnd_cands:
            if b not in books:
                rows.append({Constants.COL_USER_ID: uid, Constants.COL_BOOK_ID: b, Constants.COL_RELEVANCE: 0})
                cnt += 1
                if cnt >= n_random: break

        # Hard Negatives (Popular)
        hard_cands = np.random.choice(top_books, size=n_hard + 3)
        cnt = 0
        for b in hard_cands:
            if b not in books:
                rows.append({Constants.COL_USER_ID: uid, Constants.COL_BOOK_ID: b, Constants.COL_RELEVANCE: 0})
                cnt += 1
                if cnt >= n_hard: break

    return pd.concat([train_df, pd.DataFrame(rows)], ignore_index=True)


def clean_data_for_catboost(df, cat_cols):
    obj_cols = df.select_dtypes(include=['object']).columns
    garbage_cols = [c for c in obj_cols if c not in cat_cols]
    if garbage_cols: df = df.drop(columns=garbage_cols)
    for c in cat_cols:
        if c in df.columns: df[c] = df[c].astype(str)
    return df


def expand_candidates(df):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r[Constants.COL_BOOK_ID_LIST]): continue
        for b in str(r[Constants.COL_BOOK_ID_LIST]).split(','):
            if b.strip(): rows.append((r[Constants.COL_USER_ID], int(b.strip())))
    return pd.DataFrame(rows, columns=[Constants.COL_USER_ID, Constants.COL_BOOK_ID])


# === MAIN ===
def main():
    seed_everything()

    train_df, cand_df, u_meta, b_meta, desc_df = load_and_prep()
    bert_embs = compute_bert_embeddings(desc_df)

    # Pre-calculate popularity for hard negatives
    pop_counts = train_df[Constants.COL_BOOK_ID].value_counts()
    top_500_books = pop_counts.head(500).index.values

    # --- STEP 1: Validation ---
    print("\n" + "=" * 30 + " PHASE 1: VALIDATION " + "=" * 30)
    train_df = train_df.sort_values(Constants.COL_TIMESTAMP)
    split_idx = int(len(train_df) * (1 - Config.VAL_SIZE_RATIO))
    train_part = train_df.iloc[:split_idx].copy()
    val_part = train_df.iloc[split_idx:].copy()

    svd_data_val = train_svd_model(train_part)
    u_prof_val, _ = calculate_user_bert_profiles(train_part, bert_embs)

    # Mixed Negatives for Validation
    all_books = b_meta[Constants.COL_BOOK_ID].unique()
    train_part_neg = generate_negatives(train_part, all_books, top_500_books)
    val_part_neg = generate_negatives(val_part, all_books, top_500_books)

    train_feat, cat_cols = build_features(train_part_neg, u_meta, b_meta, desc_df, svd_data_val, bert_embs, train_part,
                                          u_prof_val)
    val_feat, _ = build_features(val_part_neg, u_meta, b_meta, desc_df, svd_data_val, bert_embs, train_part, u_prof_val)

    print("Sorting by groups...")
    train_feat = train_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)
    val_feat = val_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)

    drop_cols = [Constants.COL_USER_ID, Constants.COL_BOOK_ID, Constants.COL_RELEVANCE, Constants.COL_HAS_READ,
                 Constants.COL_TIMESTAMP,
                 "description", "als_weight", "weight", "title", "author_name", "image_url", "book_id_list"]

    X_tr = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns], errors='ignore')
    X_val = val_feat.drop(columns=[c for c in drop_cols if c in val_feat.columns], errors='ignore')
    real_cats = [c for c in cat_cols if c in X_tr.columns]
    X_tr = clean_data_for_catboost(X_tr, real_cats)
    X_val = clean_data_for_catboost(X_val, real_cats)

    train_pool = Pool(data=X_tr, label=train_feat[Constants.COL_RELEVANCE], group_id=train_feat[Constants.COL_USER_ID],
                      cat_features=real_cats)
    val_pool = Pool(data=X_val, label=val_feat[Constants.COL_RELEVANCE], group_id=val_feat[Constants.COL_USER_ID],
                    cat_features=real_cats)

    model = CatBoostRanker(**Config.CB_PARAMS)
    model.fit(train_pool, eval_set=val_pool)
    best_iter = model.best_iteration_
    print(f">>> Best Iteration: {best_iter}")

    del train_part, val_part, train_feat, val_feat, train_pool, val_pool, model, svd_data_val
    gc.collect()

    # --- STEP 2: ENSEMBLE ---
    print("\n" + "=" * 30 + " ENSEMBLE TRAINING " + "=" * 30)

    svd_data_full = train_svd_model(train_df)
    u_prof_full, _ = calculate_user_bert_profiles(train_df, bert_embs)

    # Mixed Negatives for Full Train
    train_full_neg = generate_negatives(train_df, all_books, top_500_books)
    train_full_feat, _ = build_features(train_full_neg, u_meta, b_meta, desc_df, svd_data_full, bert_embs, train_df,
                                        u_prof_full)

    train_full_feat = train_full_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)
    X_full = train_full_feat.drop(columns=[c for c in drop_cols if c in train_full_feat.columns], errors='ignore')
    X_full = clean_data_for_catboost(X_full, real_cats)
    full_pool = Pool(data=X_full, label=train_full_feat[Constants.COL_RELEVANCE],
                     group_id=train_full_feat[Constants.COL_USER_ID], cat_features=real_cats)

    # TEST Data
    cand_exp = expand_candidates(cand_df)
    cand_feat, _ = build_features(cand_exp, u_meta, b_meta, desc_df, svd_data_full, bert_embs, train_df, u_prof_full)
    X_test = cand_feat.drop(columns=[Constants.COL_USER_ID, Constants.COL_BOOK_ID], errors='ignore')
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors='ignore')
    for f in X_full.columns:
        if f not in X_test.columns: X_test[f] = 0
    X_test = X_test[X_full.columns]
    X_test = clean_data_for_catboost(X_test, real_cats)

    # Training Loop
    final_scores = np.zeros(len(X_test))

    for i, seed in enumerate(Config.SEEDS):
        print(f"Training Model {i + 1}/{len(Config.SEEDS)} | Seed: {seed} ...")

        params = Config.CB_PARAMS.copy()
        params['iterations'] = best_iter
        params['random_seed'] = seed

        # Меняем глубину для разнообразия (чередуем 6 и 8)
        if i % 2 == 1:
            params['depth'] = 8
        else:
            params['depth'] = 6

        m = CatBoostRanker(**params)
        m.fit(full_pool)

        final_scores += m.predict(X_test)
        m.save_model(Config.MODEL_DIR / f"catboost_final_seed_{seed}.cbm")
        gc.collect()

    cand_feat['score'] = final_scores / len(Config.SEEDS)

    # Submit
    cand_feat = cand_feat.sort_values([Constants.COL_USER_ID, 'score'], ascending=[True, False])
    top_20 = cand_feat.groupby(Constants.COL_USER_ID).head(20)
    sub = top_20.groupby(Constants.COL_USER_ID)[Constants.COL_BOOK_ID].apply(
        lambda x: ",".join(map(str, x))).reset_index()
    sub.columns = [Constants.COL_USER_ID, Constants.COL_BOOK_ID_LIST]

    targets = pd.read_csv(Config.RAW_DATA_DIR / Constants.TARGETS_FILENAME)
    final_sub = targets.merge(sub, on=Constants.COL_USER_ID, how='left').fillna("")

    out = Config.SUBMISSION_DIR / "submission_v7_final.csv"
    final_sub.to_csv(out, index=False)
    print(f"Final Ensemble Done! Saved to {out}")


if __name__ == "__main__":
    main()