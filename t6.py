# ... (предыдущий код без изменений) ...
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

# SVD
import scipy.sparse as sparse
from sklearn.decomposition import TruncatedSVD

# Torch/Transformers
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("WARNING: BERT not available")

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

    RANDOM_STATE = 42
    NEGATIVES_PER_USER = 7  # Чуть меньше, так как данных станет больше за счет фичей

    # BERT
    USE_BERT = True
    BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
    BERT_BATCH_SIZE = 8  # Можно увеличить до 16/32 если RAM позволяет
    BERT_MAX_LEN = 128

    # Temporal Split
    VAL_SIZE_RATIO = 0.2

    # CatBoost
    CB_PARAMS = {
        'loss_function': 'YetiRank',
        'iterations': 3000,  # Поставим с запасом, early_stopping остановит
        'learning_rate': 0.03,  # Чуть медленнее для точности
        'depth': 6,  # Оптимально для CPU/GPU
        'task_type': 'CPU',  # i9 справится отлично с 64GB RAM
        'verbose': 100,
        'random_seed': RANDOM_STATE,
        'eval_metric': 'NDCG:top=20',
        'early_stopping_rounds': 200
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

    # Определяем девайс для BERT (тут GPU пригодится для инференса)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running BERT on {device}")

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
    print("Computing BERT embeddings...")
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
    svd = TruncatedSVD(n_components=64, random_state=Config.RANDOM_STATE)  # Увеличили до 64 компонент

    user_factors = svd.fit_transform(sparse_matrix)
    item_factors = svd.components_.T
    return user_factors, item_factors, user_map, book_map


def get_svd_score(user_ids, book_ids, u_fac, i_fac, u_map, b_map):
    u_indices = np.array([u_map.get(u, -1) for u in user_ids])
    b_indices = np.array([b_map.get(b, -1) for b in book_ids])
    scores = np.zeros(len(user_ids), dtype=np.float32)
    mask = (u_indices != -1) & (b_indices != -1)
    if mask.sum() > 0:
        scores[mask] = np.sum(u_fac[u_indices[mask]] * i_fac[b_indices[mask]], axis=1)
    return scores


# --- HEURISTIC FEATURES ---
def add_heuristic_features(df, train_history_df, book_meta):
    """
    Добавляет фичи на основе совпадений.
    train_history_df: полный исторический датасет (чтобы знать предпочтения юзера)
    """
    print("Generating heuristic features...")

    # 1. Book Popularity (Global)
    book_stats = train_history_df.groupby(Constants.COL_BOOK_ID).agg(
        book_pop_count=(Constants.COL_USER_ID, 'count'),
        book_global_mean=(Constants.COL_HAS_READ, 'mean')  # % прочтений vs планов
    ).reset_index()

    # 2. Author Stats
    # Получаем author_id для train_history
    train_with_meta = train_history_df.merge(book_meta[[Constants.COL_BOOK_ID, 'author_id']], on=Constants.COL_BOOK_ID,
                                             how='left')


    user_author_stats = train_with_meta.groupby([Constants.COL_USER_ID, 'author_id'])[Constants.COL_HAS_READ].agg(
        user_author_count='count',
        user_author_mean='mean'
    ).reset_index()

    df = df.merge(book_stats, on=Constants.COL_BOOK_ID, how='left')

    if 'author_id' not in df.columns:
        df = df.merge(book_meta[[Constants.COL_BOOK_ID, 'author_id']], on=Constants.COL_BOOK_ID, how='left')

    df = df.merge(user_author_stats, on=[Constants.COL_USER_ID, 'author_id'], how='left')

    df['book_pop_count'] = df['book_pop_count'].fillna(0)
    df['book_global_mean'] = df['book_global_mean'].fillna(train_history_df[Constants.COL_HAS_READ].mean())
    df['user_author_count'] = df['user_author_count'].fillna(0)
    df['user_author_mean'] = df['user_author_mean'].fillna(0)

    return df


# --- CORE LOGIC ---
def load_and_prep():
    print("Loading data...")
    dtype_spec = {Constants.COL_USER_ID: "int32", Constants.COL_BOOK_ID: "int32", Constants.COL_HAS_READ: "int32"}
    train = pd.read_csv(Config.RAW_DATA_DIR / Constants.TRAIN_FILENAME, dtype=dtype_spec,
                        parse_dates=[Constants.COL_TIMESTAMP])

    train[Constants.COL_RELEVANCE] = train[Constants.COL_HAS_READ].map({1: 2, 0: 1}).astype("int8")

    candidates = pd.read_csv(Config.RAW_DATA_DIR / Constants.CANDIDATES_FILENAME,
                             dtype={Constants.COL_USER_ID: "int32"})
    user_meta = pd.read_csv(Config.RAW_DATA_DIR / Constants.USER_DATA_FILENAME)
    book_meta = pd.read_csv(Config.RAW_DATA_DIR / Constants.BOOK_DATA_FILENAME).drop_duplicates(Constants.COL_BOOK_ID)
    book_desc = pd.read_csv(Config.RAW_DATA_DIR / Constants.BOOK_DESCRIPTIONS_FILENAME)

    return train, candidates, user_meta, book_meta, book_desc


def generate_negatives(train_df, all_books):
    print("Generating negatives...")
    user_inter = train_df.groupby(Constants.COL_USER_ID)[Constants.COL_BOOK_ID].apply(set).to_dict()
    all_books_arr = np.array(all_books)
    rows = []
    for uid, books in tqdm(user_inter.items()):
        cands = np.random.choice(all_books_arr, size=Config.NEGATIVES_PER_USER + 5)
        cnt = 0
        for b in cands:
            if b not in books:
                rows.append({Constants.COL_USER_ID: uid, Constants.COL_BOOK_ID: b, Constants.COL_RELEVANCE: 0})
                cnt += 1
                if cnt >= Config.NEGATIVES_PER_USER: break
    return pd.concat([train_df, pd.DataFrame(rows)], ignore_index=True)


def build_features(df, u_meta, b_meta, desc_df, svd_data, bert_embs, train_history_full):
    print("Building features...")
    df = df.merge(u_meta, on=Constants.COL_USER_ID, how='left')
    # Merge book meta carefully
    cols_to_use = [c for c in b_meta.columns if c not in df.columns or c == Constants.COL_BOOK_ID]
    df = df.merge(b_meta[cols_to_use], on=Constants.COL_BOOK_ID, how='left')

    # SVD
    u_fac, i_fac, u_map, b_map = svd_data
    df[Constants.F_SVD_SCORE] = get_svd_score(df[Constants.COL_USER_ID], df[Constants.COL_BOOK_ID], u_fac, i_fac, u_map,
                                              b_map)

    df = add_heuristic_features(df, train_history_full, b_meta)

    # BERT (FULL 768)
    if bert_embs:
        print("Attaching FULL BERT embeddings...")
        # Определяем размерность автоматически по первому вектору
        sample_key = next(iter(bert_embs))
        dim = len(bert_embs[sample_key])  # Обычно 768

        emb_matrix = np.zeros((len(df), dim), dtype=np.float32)

        book_ids = df[Constants.COL_BOOK_ID].values
        # Быстрый маппинг
        for i, bid in enumerate(tqdm(book_ids, desc="BERT map")):
            if bid in bert_embs:
                emb_matrix[i] = bert_embs[bid]  # Берем целиком

        bert_df = pd.DataFrame(emb_matrix, columns=[f"bert_{i}" for i in range(dim)], index=df.index)
        df = pd.concat([df, bert_df], axis=1)

    # Clean types
    cat_cols = ['gender', 'author_id', 'publisher', 'language']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unk").astype(str)

    num_cols = ['age', 'publication_year', 'avg_rating', 'book_pop_count', 'user_author_count']
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df, cat_cols

def expand_candidates(df):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r[Constants.COL_BOOK_ID_LIST]): continue
        for b in str(r[Constants.COL_BOOK_ID_LIST]).split(','):
            if b.strip(): rows.append((r[Constants.COL_USER_ID], int(b.strip())))
    return pd.DataFrame(rows, columns=[Constants.COL_USER_ID, Constants.COL_BOOK_ID])


def clean_data_for_catboost(df, cat_cols):
    """
    Удаляет лишние текстовые колонки, которые не заявлены как категории.
    Приводит категории к строкам.
    """
    obj_cols = df.select_dtypes(include=['object']).columns
    garbage_cols = [c for c in obj_cols if c not in cat_cols]

    if garbage_cols:
        print(f"!!! WARNING: Dropping accidental text columns: {garbage_cols}")
        df = df.drop(columns=garbage_cols)

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


def main():
    seed_everything()

    train_df, cand_df, u_meta, b_meta, desc_df = load_and_prep()

    bert_embs = compute_bert_embeddings(desc_df)

    # === ЭТАП 1: ЧЕСТНАЯ ВАЛИДАЦИЯ ===
    print("\n" + "=" * 30)
    print("PHASE 1: HONEST VALIDATION")
    print("=" * 30)

    train_df = train_df.sort_values(Constants.COL_TIMESTAMP)
    split_idx = int(len(train_df) * (1 - Config.VAL_SIZE_RATIO))

    train_part = train_df.iloc[:split_idx].copy()
    val_part = train_df.iloc[split_idx:].copy()

    print(f"Train part: {len(train_part)}, Val part: {len(val_part)}")

    # Обучаем SVD *ТОЛЬКО* на train_part
    print("Training SVD on Train Part (No Leak)...")
    svd_data_val = train_svd_model(train_part)

    all_books = b_meta[Constants.COL_BOOK_ID].unique()
    train_part_neg = generate_negatives(train_part, all_books)
    val_part_neg = generate_negatives(val_part, all_books)


    print("Building features for validation...")
    train_feat, cat_cols = build_features(train_part_neg, u_meta, b_meta, desc_df, svd_data_val, bert_embs, train_part)
    val_feat, _ = build_features(val_part_neg, u_meta, b_meta, desc_df, svd_data_val, bert_embs, train_part)

    drop_cols = [
        Constants.COL_USER_ID, Constants.COL_BOOK_ID, Constants.COL_RELEVANCE,
        Constants.COL_HAS_READ, Constants.COL_TIMESTAMP,
        "description", "als_weight", "weight",
        "title", "author_name", "image_url", "book_id_list"
    ]

    print("Sorting by groups...")
    train_feat = train_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)
    val_feat = val_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)

    X_tr = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns], errors='ignore')
    y_tr = train_feat[Constants.COL_RELEVANCE]
    g_tr = train_feat[Constants.COL_USER_ID]

    X_val = val_feat.drop(columns=[c for c in drop_cols if c in val_feat.columns], errors='ignore')
    y_val = val_feat[Constants.COL_RELEVANCE]
    g_val = val_feat[Constants.COL_USER_ID]

    # Чистка типов (Fix ошибки CatBoost)
    real_cats = [c for c in cat_cols if c in X_tr.columns]
    X_tr = clean_data_for_catboost(X_tr, real_cats)
    X_val = clean_data_for_catboost(X_val, real_cats)

    print(f"Features: {len(X_tr.columns)}")

    train_pool = Pool(data=X_tr, label=y_tr, group_id=g_tr, cat_features=real_cats)
    val_pool = Pool(data=X_val, label=y_val, group_id=g_val, cat_features=real_cats)

    # Обучение с валидацией
    model = CatBoostRanker(**Config.CB_PARAMS)
    model.fit(train_pool, eval_set=val_pool)

    # Запоминаем лучшее число итераций!
    best_iter = model.best_iteration_
    print(f"\n>>> REAL Best Iteration found: {best_iter} <<<\n")

    # Чистим память перед рефитом
    del train_part, val_part, train_feat, val_feat, X_tr, y_tr, X_val, y_val, train_pool, val_pool, model, svd_data_val
    gc.collect()

    # === ЭТАП 2: REFIT НА ПОЛНЫХ ДАННЫХ ===
    print("\n" + "=" * 30)
    print("PHASE 2: REFIT (Full Train)")
    print("=" * 30)

    # Теперь учим SVD на ВСЕМ train_df
    print("Training SVD on Full Data...")
    svd_data_full = train_svd_model(train_df)

    # Генерируем негативы для всего трейна
    train_full_neg = generate_negatives(train_df, all_books)

    # Фичи на основе полной истории
    train_full_feat, _ = build_features(train_full_neg, u_meta, b_meta, desc_df, svd_data_full, bert_embs, train_df)

    train_full_feat = train_full_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)

    X_full = train_full_feat.drop(columns=[c for c in drop_cols if c in train_full_feat.columns], errors='ignore')
    X_full = clean_data_for_catboost(X_full, real_cats)

    y_full = train_full_feat[Constants.COL_RELEVANCE]
    g_full = train_full_feat[Constants.COL_USER_ID]

    print(f"Refitting with {best_iter} trees on full data...")
    refit_params = Config.CB_PARAMS.copy()
    refit_params['iterations'] = best_iter
    # Важно: убираем use_best_model=True, так как нет eval_set

    final_model = CatBoostRanker(**refit_params)
    full_pool = Pool(data=X_full, label=y_full, group_id=g_full, cat_features=real_cats)
    final_model.fit(full_pool)

    final_model.save_model(Config.MODEL_DIR / "catboost_refit_v4.cbm")

    # === ЭТАП 3: INFERENCE ===
    print("\n" + "=" * 30)
    print("PHASE 3: INFERENCE")
    print("=" * 30)

    cand_exp = expand_candidates(cand_df)
    # Фичи для кандидатов строим на основе SVD_FULL и TRAIN_DF (полной истории)
    cand_feat, _ = build_features(cand_exp, u_meta, b_meta, desc_df, svd_data_full, bert_embs, train_df)

    X_test = cand_feat.drop(columns=[Constants.COL_USER_ID, Constants.COL_BOOK_ID], errors='ignore')
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors='ignore')

    # Выравнивание
    for f in final_model.feature_names_:
        if f not in X_test.columns: X_test[f] = 0
    X_test = X_test[final_model.feature_names_]

    X_test = clean_data_for_catboost(X_test, real_cats)

    scores = final_model.predict(X_test)
    cand_feat['score'] = scores

    # Submit
    cand_feat = cand_feat.sort_values([Constants.COL_USER_ID, 'score'], ascending=[True, False])
    top_20 = cand_feat.groupby(Constants.COL_USER_ID).head(20)
    sub = top_20.groupby(Constants.COL_USER_ID)[Constants.COL_BOOK_ID].apply(
        lambda x: ",".join(map(str, x))).reset_index()
    sub.columns = [Constants.COL_USER_ID, Constants.COL_BOOK_ID_LIST]

    targets = pd.read_csv(Config.RAW_DATA_DIR / Constants.TARGETS_FILENAME)
    final_sub = targets.merge(sub, on=Constants.COL_USER_ID, how='left').fillna("")

    out_path = Config.SUBMISSION_DIR / "submission_v4_honest.csv"
    final_sub.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()