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
from sklearn.metrics.pairwise import cosine_similarity

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
    # Чуть увеличим негативы, раз модель так быстро учится
    NEGATIVES_PER_USER = 15

    # BERT
    USE_BERT = True
    BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
    BERT_BATCH_SIZE = 8
    BERT_MAX_LEN = 128

    VAL_SIZE_RATIO = 0.2

    CB_PARAMS = {
        'loss_function': 'YetiRank',
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 6,
        'task_type': 'CPU',
        'thread_count': -1,
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

    # КЭШ: Если файл есть, грузим его. Если переименовал в прошлый раз - используй то имя.
    cache_path = Config.PROCESSED_DATA_DIR / "bert_embeddings_full.pkl"
    if cache_path.exists():
        print("Loading cached BERT embeddings...")
        return joblib.load(cache_path)

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
                # Mean Pooling
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
    svd = TruncatedSVD(n_components=64, random_state=Config.RANDOM_STATE)

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


# --- FEATURES ---

def add_heuristic_features(df, train_history_df, book_meta):
    print("Generating heuristic features...")
    # Book Popularity
    book_stats = train_history_df.groupby(Constants.COL_BOOK_ID).agg(
        book_pop_count=(Constants.COL_USER_ID, 'count'),
        book_global_mean=(Constants.COL_HAS_READ, 'mean')
    ).reset_index()

    # Author Stats
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


def calculate_user_bert_profiles(train_history_df, bert_embs):
    """
    Считает средний BERT-вектор для каждого пользователя на основе прочитанных книг.
    Возвращает dict {user_id: np.array}
    """
    print("Calculating User BERT Profiles...")

    # Конвертируем bert_embs словарь в DataFrame для быстрого мерджа
    # (Можно оптимизировать, но для 64ГБ RAM сойдет)
    bert_data = []
    # Берем размерность
    sample_key = next(iter(bert_embs))
    dim = len(bert_embs[sample_key])

    for bid, vec in bert_embs.items():
        bert_data.append([bid] + list(vec))

    bert_cols = [f"b{i}" for i in range(dim)]
    bert_df = pd.DataFrame(bert_data, columns=[Constants.COL_BOOK_ID] + bert_cols)

    # Мержим историю с эмбеддингами
    # Берем только позитивные взаимодействия для профиля (has_read=1 или 0)
    # Можно фильтровать только has_read=1, но 0 (планы) тоже интерес
    merged = train_history_df.merge(bert_df, on=Constants.COL_BOOK_ID, how='inner')

    # Группируем по юзеру и берем среднее
    user_profiles = merged.groupby(Constants.COL_USER_ID)[bert_cols].mean()

    # Превращаем в словарь для быстрого доступа
    # Это будет dict {user_id: array([x, y, z...])}
    profile_map = {uid: row.values for uid, row in user_profiles.iterrows()}
    return profile_map, dim


def add_semantic_features(df, user_profiles, bert_embs, dim):
    """
    Считает косинусную близость между профилем юзера и книгой.
    """
    print("Calculating Semantic Similarity (User <-> Book)...")

    # Подготовим массивы
    # Если профиля нет (новый юзер) - вектор нулей
    # Если книги нет (нет эмбеддинга) - вектор нулей

    user_vecs = np.zeros((len(df), dim), dtype=np.float32)
    book_vecs = np.zeros((len(df), dim), dtype=np.float32)

    user_ids = df[Constants.COL_USER_ID].values
    book_ids = df[Constants.COL_BOOK_ID].values

    # Заполняем (можно ускорить, но цикл с tqdm нагляден)
    # Для продакшена лучше pandas apply или map

    # Векторизированный map через reindexing работает быстрее циклов
    # 1. Book Vectors
    # Создаем матрицу всех книг
    # Но у нас book_ids могут повторяться.
    # Проще через map, если bert_embs это dict

    # Оптимизация:
    # Делаем функцию-геттер с дефолтом
    def get_u(uid): return user_profiles.get(uid, np.zeros(dim))

    def get_b(bid): return bert_embs.get(bid, np.zeros(dim))

    # Это может быть медленно на миллионах строк, но у нас сотни тысяч.
    # Попробуем списковое включение, оно быстрее apply
    user_vecs = np.array([get_u(u) for u in user_ids])
    book_vecs = np.array([get_b(b) for b in book_ids])

    # Считаем косинус руками (быстрее, чем sklearn на больших батчах)
    # CosSim(A, B) = Dot(A, B) / (Norm(A) * Norm(B))

    dot_product = np.sum(user_vecs * book_vecs, axis=1)
    norm_u = np.linalg.norm(user_vecs, axis=1)
    norm_b = np.linalg.norm(book_vecs, axis=1)

    # Избегаем деления на ноль
    similarity = dot_product / (norm_u * norm_b + 1e-9)

    df[Constants.F_BERT_SIM] = similarity.astype(np.float32)
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


def build_features(df, u_meta, b_meta, desc_df, svd_data, bert_embs, train_history_full, user_profiles=None):
    print("Building features...")
    df = df.merge(u_meta, on=Constants.COL_USER_ID, how='left')
    cols_to_use = [c for c in b_meta.columns if c not in df.columns or c == Constants.COL_BOOK_ID]
    df = df.merge(b_meta[cols_to_use], on=Constants.COL_BOOK_ID, how='left')

    # SVD
    u_fac, i_fac, u_map, b_map = svd_data
    df[Constants.F_SVD_SCORE] = get_svd_score(df[Constants.COL_USER_ID], df[Constants.COL_BOOK_ID], u_fac, i_fac, u_map,
                                              b_map)

    # Heuristics
    df = add_heuristic_features(df, train_history_full, b_meta)

    # BERT Similarity (New Feature!)
    if user_profiles and bert_embs:
        sample_key = next(iter(bert_embs))
        dim = len(bert_embs[sample_key])
        df = add_semantic_features(df, user_profiles, bert_embs, dim)

    # BERT Raw Features (Still keeping them, boosted trees like raw numbers too)
    if bert_embs:
        # Берем 32 компоненты для сырых фичей, чтобы не дублировать всё
        dim_raw = 32
        emb_matrix = np.zeros((len(df), dim_raw), dtype=np.float32)
        book_ids = df[Constants.COL_BOOK_ID].values
        for i, bid in enumerate(book_ids):
            if bid in bert_embs:
                emb_matrix[i] = bert_embs[bid][:dim_raw]
        bert_df = pd.DataFrame(emb_matrix, columns=[f"bert_{i}" for i in range(dim_raw)], index=df.index)
        df = pd.concat([df, bert_df], axis=1)

    # Clean types
    cat_cols = ['gender', 'author_id', 'publisher', 'language']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unk").astype(str)

    num_cols = ['age', 'publication_year', 'avg_rating', 'book_pop_count', 'user_author_count', Constants.F_BERT_SIM]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df, cat_cols


def clean_data_for_catboost(df, cat_cols):
    obj_cols = df.select_dtypes(include=['object']).columns
    garbage_cols = [c for c in obj_cols if c not in cat_cols]
    if garbage_cols:
        df = df.drop(columns=garbage_cols)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def expand_candidates(df):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r[Constants.COL_BOOK_ID_LIST]): continue
        for b in str(r[Constants.COL_BOOK_ID_LIST]).split(','):
            if b.strip(): rows.append((r[Constants.COL_USER_ID], int(b.strip())))
    return pd.DataFrame(rows, columns=[Constants.COL_USER_ID, Constants.COL_BOOK_ID])


def main():
    seed_everything()

    train_df, cand_df, u_meta, b_meta, desc_df = load_and_prep()

    # 4. BERT Embeddings
    bert_embs = compute_bert_embeddings(desc_df)

    # === VALIDATION ===
    print("\n" + "=" * 30 + " PHASE 1: VALIDATION " + "=" * 30)

    train_df = train_df.sort_values(Constants.COL_TIMESTAMP)
    split_idx = int(len(train_df) * (1 - Config.VAL_SIZE_RATIO))
    train_part = train_df.iloc[:split_idx].copy()
    val_part = train_df.iloc[split_idx:].copy()

    # 1. Train models ONLY on train_part
    print("Training sub-models on Train Part...")
    svd_data_val = train_svd_model(train_part)
    # Считаем профили юзеров ТОЛЬКО по истории train_part
    user_profiles_val, _ = calculate_user_bert_profiles(train_part, bert_embs)

    # 2. Negatives
    all_books = b_meta[Constants.COL_BOOK_ID].unique()
    train_part_neg = generate_negatives(train_part, all_books)
    val_part_neg = generate_negatives(val_part, all_books)

    # 3. Features
    # Важно: передаем user_profiles_val
    train_feat, cat_cols = build_features(train_part_neg, u_meta, b_meta, desc_df, svd_data_val, bert_embs, train_part,
                                          user_profiles_val)
    val_feat, _ = build_features(val_part_neg, u_meta, b_meta, desc_df, svd_data_val, bert_embs, train_part,
                                 user_profiles_val)

    # 4. Clean & Sort
    print("Sorting by groups...")
    train_feat = train_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)
    val_feat = val_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)

    drop_cols = [Constants.COL_USER_ID, Constants.COL_BOOK_ID, Constants.COL_RELEVANCE, Constants.COL_HAS_READ,
                 Constants.COL_TIMESTAMP,
                 "description", "als_weight", "weight", "title", "author_name", "image_url", "book_id_list"]

    X_tr = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns], errors='ignore')
    y_tr = train_feat[Constants.COL_RELEVANCE]
    g_tr = train_feat[Constants.COL_USER_ID]

    X_val = val_feat.drop(columns=[c for c in drop_cols if c in val_feat.columns], errors='ignore')
    y_val = val_feat[Constants.COL_RELEVANCE]
    g_val = val_feat[Constants.COL_USER_ID]

    real_cats = [c for c in cat_cols if c in X_tr.columns]
    X_tr = clean_data_for_catboost(X_tr, real_cats)
    X_val = clean_data_for_catboost(X_val, real_cats)

    print(f"Features: {len(X_tr.columns)}")

    train_pool = Pool(data=X_tr, label=y_tr, group_id=g_tr, cat_features=real_cats)
    val_pool = Pool(data=X_val, label=y_val, group_id=g_val, cat_features=real_cats)

    model = CatBoostRanker(**Config.CB_PARAMS)
    model.fit(train_pool, eval_set=val_pool)
    best_iter = model.best_iteration_
    print(f"\n>>> Best Iteration: {best_iter} <<<\n")

    # Cleanup
    del train_part, val_part, train_feat, val_feat, X_tr, y_tr, X_val, y_val, train_pool, val_pool, model, svd_data_val, user_profiles_val
    gc.collect()

    # === REFIT ===
    print("\n" + "=" * 30 + " PHASE 2: REFIT " + "=" * 30)

    # Учим модели на ВСЕМ train
    svd_data_full = train_svd_model(train_df)
    user_profiles_full, _ = calculate_user_bert_profiles(train_df, bert_embs)

    train_full_neg = generate_negatives(train_df, all_books)
    train_full_feat, _ = build_features(train_full_neg, u_meta, b_meta, desc_df, svd_data_full, bert_embs, train_df,
                                        user_profiles_full)

    train_full_feat = train_full_feat.sort_values(Constants.COL_USER_ID).reset_index(drop=True)
    X_full = train_full_feat.drop(columns=[c for c in drop_cols if c in train_full_feat.columns], errors='ignore')
    X_full = clean_data_for_catboost(X_full, real_cats)
    y_full = train_full_feat[Constants.COL_RELEVANCE]
    g_full = train_full_feat[Constants.COL_USER_ID]

    print(f"Refitting with {best_iter} trees...")
    refit_params = Config.CB_PARAMS.copy()
    refit_params['iterations'] = best_iter

    final_model = CatBoostRanker(**refit_params)
    full_pool = Pool(data=X_full, label=y_full, group_id=g_full, cat_features=real_cats)
    final_model.fit(full_pool)
    final_model.save_model(Config.MODEL_DIR / "catboost_refit_v5.cbm")

    # === INFERENCE ===
    print("\n" + "=" * 30 + " PHASE 3: INFERENCE " + "=" * 30)
    cand_exp = expand_candidates(cand_df)
    # Фичи на основе ПОЛНЫХ моделей
    cand_feat, _ = build_features(cand_exp, u_meta, b_meta, desc_df, svd_data_full, bert_embs, train_df,
                                  user_profiles_full)

    X_test = cand_feat.drop(columns=[c for c in drop_cols if c in cand_feat.columns], errors='ignore')
    for f in final_model.feature_names_:
        if f not in X_test.columns: X_test[f] = 0
    X_test = X_test[final_model.feature_names_]
    X_test = clean_data_for_catboost(X_test, real_cats)

    scores = final_model.predict(X_test)
    cand_feat['score'] = scores

    cand_feat = cand_feat.sort_values([Constants.COL_USER_ID, 'score'], ascending=[True, False])
    top_20 = cand_feat.groupby(Constants.COL_USER_ID).head(20)
    sub = top_20.groupby(Constants.COL_USER_ID)[Constants.COL_BOOK_ID].apply(
        lambda x: ",".join(map(str, x))).reset_index()
    sub.columns = [Constants.COL_USER_ID, Constants.COL_BOOK_ID_LIST]

    targets = pd.read_csv(Config.RAW_DATA_DIR / Constants.TARGETS_FILENAME)
    final_sub = targets.merge(sub, on=Constants.COL_USER_ID, how='left').fillna("")

    final_sub.to_csv(Config.SUBMISSION_DIR / "submission_v5_semantic.csv", index=False)
    print("Done!")


if __name__ == "__main__":
    main()