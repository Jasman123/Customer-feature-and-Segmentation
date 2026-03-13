"""
=============================================================================
DEMOGRAPHIC PROXY FEATURE ENGINEERING
Based on actual MCC codes from mcc_gender_predictions.csv
Features: Age, Income Level, Education, Home Location, Work Location
=============================================================================
"""

import pandas as pd
import numpy as np

# =============================================================================
# STEP 0 — SETUP & LOAD DATA
# =============================================================================

# Load your datasets
df  = pd.read_csv('transactions.csv')          # ← change path
mcc = pd.read_csv('mcc_gender_predictions.csv')

# Ensure correct types
df['transaction_date']      = pd.to_datetime(df['transaction_date'])
df['merchant_category_id']  = df['merchant_category_id'].astype(int)
mcc['mcc_code']             = mcc['mcc_code'].astype(int)

# Filter to completed transactions only
if 'transaction_status' in df.columns:
    df = df[df['transaction_status'].str.lower() == 'completed'].copy()

# Extract time features (shared across all proxies)
df['hour']        = df['transaction_date'].dt.hour
df['day_of_week'] = df['transaction_date'].dt.dayofweek   # 0=Mon, 6=Sun
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
df['is_weekday']  = (~df['day_of_week'].isin([5, 6])).astype(int)

print("✅ Data loaded.")
print(f"   Transactions : {len(df):,}")
print(f"   Unique users : {df['user_id'].nunique():,}")
print(f"   Date range   : {df['transaction_date'].min().date()} → {df['transaction_date'].max().date()}")


# =============================================================================
# MCC LOOKUP TABLES — extracted from your actual mcc_gender_predictions.csv
# =============================================================================

# ── AGE: YOUNG signals (18–30)
# Fast food, gaming, digital goods, music, sports, schools, candy, streaming
MCC_AGE_YOUNG = [
    # Fast Food
    5814, 1083, 5414,
    # Gaming & Video Games
    5818, 5823, 5824, 5826, 5854,
    3639, 3920, 3921, 3923, 3943, 3947,
    6284,
    # Digital Goods (streaming, apps, media)
    5817, 3975, 6460, 2748, 2743, 2744, 5849, 5844, 5864,
    # Music
    5733, 573,
    # Sports & Recreation
    5356, 5941, 5553, 5655, 5650, 714,
    # Schools / Education (students)
    821, 822, 823, 824, 825, 828,
    # Candy & Confectionery
    5471, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489,
    5440, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5449,
    5405, 5406, 5407, 5408, 5415, 5416, 5418, 5456, 5457, 5473,
    5476, 5477, 5478, 5477, 544,
    # Video Rental
    3864, 3694,
    # Toy & Hobby
    5945, 5268,
]

# ── AGE: MATURE signals (45+)
# Insurance, medical, hospitals, mortgage, pension, funeral, real estate
MCC_AGE_MATURE = [
    # Hospitals
    808, 809, 3690, 3691, 3692, 3693, 3695, 3696, 3697, 3698,
    # Medical Services
    3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809,
    3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819,
    3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829,
    3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839,
    3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849,
    3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859,
    3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879,
    3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3889, 3890,
    3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899,
    883, 805,
    # Medical Labs
    807, 6322, 6323,
    # Insurance (all 63xx range — very mature audience)
    6300, 6301, 6305, 6306, 6307, 6308, 6309, 6310,
    6311, 6312, 6313, 6314, 6315, 6316, 6317, 6318, 6319,
    6320, 6321, 6324, 6325, 6326, 6327, 6328, 6329,
    6330, 6331, 6332, 6333, 6334, 6335, 6337, 6338, 6339,
    6340, 6341, 6342, 6343, 6344, 6345, 6346, 6347, 6348, 6349,
    6350, 6351, 6352, 6354, 6355, 6356, 6357, 6358, 6359,
    6361, 6364, 6365, 6366, 6367, 6368, 6369,
    6370, 6371, 6372, 6373, 6374, 6375, 6376, 6378, 6379,
    6380, 6381, 6382, 6383, 6384, 6385, 6386, 6387, 6388, 6389,
    6390, 6391, 6392, 6393, 6394, 6395, 6396, 6398, 6399,
    6405, 6407, 6412, 6413, 6414, 6417, 6419,
    # Mortgage & Real Estate
    6116, 6124, 6125, 6126, 6127, 6161, 6162, 6164, 6165,
    6166, 6167, 6168, 6169, 6176,
    6112,
    # Pension
    6371, 6372, 6378,
    # Funeral
    726,
    # Real Estate
    651,
]

# ── INCOME: HIGH signals
# Jewelry, boats, aircraft, antiques, country clubs, fur, art, cigars, hotels/resorts
MCC_INCOME_HIGH = [
    # Jewelry & Precious Stones
    5944, 5972, 5096, 5094, 5095, 5091, 5097, 5976, 5952,
    1434, 1437, 1470, 1473, 1474, 1475, 1476, 1478, 1486, 1487,
    1680, 1687, 1867,
    2144, 2146, 2147, 2178, 2567, 2637, 2643, 2653, 2657,
    2663, 2667, 2669, 2681, 2683, 2689,
    # Boats & Marine
    5563, 5551, 5557, 5558, 5565, 5591, 5518,
    4422, 4425, 4457, 4473,
    # Aircraft & Aviation
    4505, 4527, 4535, 4537, 4544, 4546, 4547, 4564, 4566,
    4592, 4597, 4740, 1161,
    # Antique Shops
    5755, 5752, 5937, 5932, 5930, 5934, 5936, 5938, 5939,
    5560, 5561, 5703, 5704, 5708, 5709, 5743, 5752, 5759,
    5760, 5761, 5762, 5763, 5769, 5779, 5831, 5836, 5837,
    5839, 5973, 5979, 765, 575,
    5324, 5327, 5342, 5347, 5360, 5394, 5396,
    # Art Galleries
    5092, 5093, 5098, 5971,
    # Country Clubs & Golf
    3906, 1483, 1226,
    # Fur Stores
    5683, 5636, 5680, 5681, 5682, 5684, 5685, 5686, 5687, 5688, 5689,
    2381,
    # Cigars
    5873, 5993, 5808,
    # Hotels & Resorts (3500-3599 range)
    3499, 3500, 3599, 3799, 3299,
    # Health & Beauty Spas
    6302, 6382, 6387,
    # Ski Resorts
    2596,
]

# ── INCOME: BUDGET signals
# Variety stores, wholesale clubs, pawn shops, used merchandise, discount stores
MCC_INCOME_BUDGET = [
    # Variety Stores (dollar stores)
    5331, 5322, 5303, 5330, 5332, 5333, 5334, 5335, 5336,
    5338, 5339, 533,
    5803,
    # Wholesale Clubs
    5300, 5301, 5302, 5306, 5470, 5957, 5815, 1464, 530,
    # Pawn Shops
    5933, 5881, 5893, 5872, 5870, 5757,
    # Used Merchandise
    5794, 5931, 5774, 5776, 5790, 5793, 5795, 5626, 5702,
    # Discount Stores
    5310, 5317, 5316, 5328,
    # Used Car Dealers
    551, 5555, 5589,
    # Convenience Stores
    5499, 5491, 5490, 5479, 5419, 1191, 1099, 1118, 1119,
    1390, 1809, 5819,
    # Fast Food (budget eating)
    5814, 1083, 5414,
]

# ── EDUCATION: HIGH signals
# Bookstores, schools, universities, computer software, stationery, publishing
MCC_EDUCATION_HIGH = [
    # Schools & Colleges
    821, 822, 823, 824, 825, 828,
    # Bookstores
    2703, 2706, 2708, 2709, 2730, 2731, 2732, 2733, 2734,
    2735, 2736, 2737, 2738, 2739, 2755, 2756, 2770, 2771,
    2772, 2775, 2778, 2780, 2783, 2786, 2788, 5942,
    # Books, Periodicals, Newspapers
    5192, 5194, 5038, 2784, 2789, 2781,
    # Digital Goods Media (books/education)
    6460, 2748, 2743, 2744, 5849, 5844, 5864,
    # Computer Software
    5734, 5751, 5758, 5738, 5746, 5754, 5788, 5180, 5181,
    5182, 5183,
    # Stationery & Office Supply
    5943, 5996, 5946, 5117,
    5111, 5112, 5113, 5116, 5806, 511,
    # Publishing & Printing
    2750, 2751, 2752, 2753, 2754, 2715, 2716, 2717, 2718, 2719,
    # Computer/Info Services
    4816, 5780, 737,
    # Tutoring & Private Instruction
    829,
    # Accounting & Professional Services
    872,
]


# =============================================================================
# STEP 1 — JOIN MCC SIGNALS TO TRANSACTIONS
# =============================================================================

def tag_mcc_signals(df, mcc_df):
    """Join MCC metadata and tag age/income/education signals per transaction."""

    # Merge MCC name for reference
    df = df.merge(
        mcc_df[['mcc_code', 'MCC_Code_name']].drop_duplicates('mcc_code'),
        left_on='merchant_category_id',
        right_on='mcc_code',
        how='left'
    )

    # Age signals
    df['is_young_mcc']  = df['merchant_category_id'].isin(MCC_AGE_YOUNG).astype(int)
    df['is_mature_mcc'] = df['merchant_category_id'].isin(MCC_AGE_MATURE).astype(int)

    # Income signals
    df['is_high_income_mcc']   = df['merchant_category_id'].isin(MCC_INCOME_HIGH).astype(int)
    df['is_budget_income_mcc'] = df['merchant_category_id'].isin(MCC_INCOME_BUDGET).astype(int)

    # Education signals
    df['is_edu_mcc'] = df['merchant_category_id'].isin(MCC_EDUCATION_HIGH).astype(int)

    print("\n✅ MCC signals tagged.")
    print(f"   Young MCC transactions   : {df['is_young_mcc'].sum():,}")
    print(f"   Mature MCC transactions  : {df['is_mature_mcc'].sum():,}")
    print(f"   High income transactions : {df['is_high_income_mcc'].sum():,}")
    print(f"   Budget transactions      : {df['is_budget_income_mcc'].sum():,}")
    print(f"   Education transactions   : {df['is_edu_mcc'].sum():,}")

    return df

df = tag_mcc_signals(df, mcc)


# =============================================================================
# STEP 2 — AGE PROXY
# =============================================================================
# Approach: Rule-based MCC signals + time-of-day + device type
# Output:   age_proxy_score (0=young, 1=mature), age_group label

def extract_device(ua):
    """Extract device type from user_agent string."""
    if pd.isna(ua):
        return 'unknown'
    ua = str(ua).lower()
    if any(x in ua for x in ['mobile', 'android', 'iphone', 'samsung']):
        return 'mobile'
    elif any(x in ua for x in ['tablet', 'ipad']):
        return 'tablet'
    else:
        return 'desktop'

df['device_type'] = df['user_agent'].apply(extract_device)

# Time-of-day age signal
df['is_late_night'] = df['hour'].between(22, 23) | df['hour'].between(0, 2)
df['is_daytime']    = df['hour'].between(9, 17)

age_features = df.groupby('user_id').agg(
    # MCC-based signals
    pct_young_mcc       = ('is_young_mcc',   'mean'),
    pct_mature_mcc      = ('is_mature_mcc',  'mean'),
    # Time signals
    pct_late_night      = ('is_late_night',  'mean'),
    pct_daytime         = ('is_daytime',     'mean'),
    # Device signals
    pct_mobile          = ('device_type',    lambda x: (x == 'mobile').mean()),
    pct_desktop         = ('device_type',    lambda x: (x == 'desktop').mean()),
    # Transaction count (older users tend to transact more steadily)
    total_transactions  = ('transaction_amount', 'count'),
).reset_index()

# Score: 0 = young, 1 = mature
# Weights based on signal reliability
age_features['age_proxy_score'] = (
    age_features['pct_mature_mcc']   * 0.35 +
    age_features['pct_daytime']      * 0.25 +
    age_features['pct_desktop']      * 0.20 +
    (1 - age_features['pct_young_mcc'])   * 0.10 +
    (1 - age_features['pct_late_night'])  * 0.10
)

def assign_age_group(score):
    if score >= 0.65:   return '45+'
    elif score >= 0.50: return '35-44'
    elif score >= 0.35: return '25-34'
    else:               return '18-24'

age_features['age_group'] = age_features['age_proxy_score'].apply(assign_age_group)

print("\n✅ Age proxy computed.")
print(age_features['age_group'].value_counts())


# =============================================================================
# STEP 3 — INCOME LEVEL
# =============================================================================
# Approach: Multi-signal scoring — spend amount + MCC tier + payment method + promo sensitivity
# Output:   income_score, income_tier (Low / Mid / High)

def payment_score(method):
    """Score payment method as income indicator (0=low, 1=high)."""
    if pd.isna(method):
        return 0.5
    method = str(method).lower()
    if any(x in method for x in ['credit', 'premium', 'gold', 'platinum']):
        return 1.0
    elif any(x in method for x in ['debit', 'prepaid']):
        return 0.4
    elif any(x in method for x in ['cod', 'cash']):
        return 0.2
    elif any(x in method for x in ['ewallet', 'digital', 'gopay', 'ovo', 'dana']):
        return 0.5
    return 0.5

df['payment_income_score'] = df['payment_method'].apply(payment_score)

income_features = df.groupby('user_id').agg(
    # Spending behavior
    avg_transaction      = ('transaction_amount', 'mean'),
    median_transaction   = ('transaction_amount', 'median'),
    total_spend          = ('transaction_amount', 'sum'),
    std_transaction      = ('transaction_amount', lambda x: x.std() if len(x) > 1 else 0),
    # MCC-based income signals
    pct_high_income_mcc  = ('is_high_income_mcc',   'mean'),
    pct_budget_mcc       = ('is_budget_income_mcc', 'mean'),
    # Payment method
    avg_payment_score    = ('payment_income_score', 'mean'),
    # Promo/discount sensitivity (inverse of income)
    pct_discounted       = ('discount_applied', 'mean'),
    avg_promo_amount     = ('promo_amount',      'mean'),
    # Loyalty (higher income = more loyalty engagement)
    uses_loyalty         = ('loyalty_program',  lambda x: x.notna().mean()),
).reset_index()

# Normalize spend to 0-1 percentile
income_features['spend_percentile'] = (
    income_features['avg_transaction'].rank(pct=True)
)

# Composite income score (0=low, 1=high)
income_features['income_score'] = (
    income_features['spend_percentile']       * 0.35 +
    income_features['avg_payment_score']      * 0.25 +
    income_features['pct_high_income_mcc']    * 0.15 +
    (1 - income_features['pct_budget_mcc'])   * 0.10 +
    (1 - income_features['pct_discounted'].fillna(0)) * 0.10 +
    income_features['uses_loyalty']           * 0.05
)

def assign_income_tier(score):
    if score >= 0.67:   return 'High'
    elif score >= 0.33: return 'Mid'
    else:               return 'Low'

income_features['income_tier'] = income_features['income_score'].apply(assign_income_tier)

print("\n✅ Income level computed.")
print(income_features['income_tier'].value_counts())


# =============================================================================
# STEP 4 — EDUCATION BACKGROUND
# =============================================================================
# Approach: MCC signals (bookstores, schools, software, stationery) + spending sophistication
# Output:   education_score, education_level (Basic / Moderate / Advanced)
# ⚠️  Note: Weak proxy — treat as low-confidence feature

# Merchant name keyword boost (supplements MCC signal)
EDU_MERCHANT_KEYWORDS = [
    'book', 'bookstore', 'library', 'university', 'college', 'school',
    'academy', 'course', 'training', 'seminar', 'workshop', 'udemy',
    'coursera', 'skillshare', 'office', 'stationery', 'lab', 'research'
]

def merchant_edu_signal(name):
    if pd.isna(name):
        return 0
    name = str(name).lower()
    return int(any(kw in name for kw in EDU_MERCHANT_KEYWORDS))

df['is_edu_merchant'] = df['merchant_name'].apply(merchant_edu_signal)

edu_features = df.groupby('user_id').agg(
    # MCC-based education signal
    pct_edu_mcc          = ('is_edu_mcc',       'mean'),
    # Merchant name signal
    pct_edu_merchant     = ('is_edu_merchant',  'mean'),
    # Sophistication proxies
    category_diversity   = ('merchant_category_id', 'nunique'),
    # Prefer quality over discount (educated = less price-sensitive)
    pct_no_discount      = ('discount_applied', lambda x: (x == 0).mean()),
    avg_merchant_rating  = ('merchant_rating',  'mean'),
).reset_index()

# Normalize category diversity
edu_features['diversity_pct'] = (
    edu_features['category_diversity'].rank(pct=True)
)

# Education score (0=low, 1=high)
edu_features['education_score'] = (
    edu_features['pct_edu_mcc']       * 0.40 +
    edu_features['pct_edu_merchant']  * 0.30 +
    edu_features['diversity_pct']     * 0.15 +
    edu_features['pct_no_discount'].fillna(0) * 0.15
)

def assign_education_level(score):
    if score >= 0.60:   return 'Advanced'
    elif score >= 0.30: return 'Moderate'
    else:               return 'Basic'

edu_features['education_level'] = edu_features['education_score'].apply(assign_education_level)

print("\n✅ Education proxy computed. ⚠️  Low confidence — use with caution.")
print(edu_features['education_level'].value_counts())


# =============================================================================
# STEP 5 — HOME LOCATION
# =============================================================================
# Approach: Most frequent geo during weekend + evening hours
# Output:   home_location, home_location_confidence

# Home time window: weekends OR evenings (18:00–23:00)
df['is_home_time'] = (
    (df['is_weekend'] == 1) |
    df['hour'].between(18, 23)
).astype(int)

home_df = df[df['is_home_time'] == 1]

def most_frequent_geo(series):
    """Return most frequent value, or None if empty."""
    counts = series.value_counts()
    return counts.index[0] if len(counts) > 0 else None

def geo_confidence(series):
    """How dominant is the top geo? (top_count / total)"""
    counts = series.value_counts()
    if len(counts) == 0:
        return 0.0
    return counts.iloc[0] / len(series)

home_location = home_df.groupby('user_id')['geo_location'].agg([
    ('home_location',            most_frequent_geo),
    ('home_location_confidence', geo_confidence),
    ('home_txn_count',           'count'),
]).reset_index()

# Flag low-confidence home locations
home_location['home_location_reliable'] = (
    (home_location['home_location_confidence'] >= 0.4) &
    (home_location['home_txn_count'] >= 3)
).astype(int)

print("\n✅ Home location computed.")
print(f"   Reliable home locations : {home_location['home_location_reliable'].sum():,} users")
print(f"   Avg confidence          : {home_location['home_location_confidence'].mean():.2f}")


# =============================================================================
# STEP 6 — WORK LOCATION
# =============================================================================
# Approach: Most frequent geo during weekday business hours (09:00–17:00)
#           Cross-check against home location to validate it's different
# Output:   work_location, is_remote_worker flag

df['is_work_time'] = (
    (df['is_weekday'] == 1) &
    df['hour'].between(9, 17)
).astype(int)

work_df = df[df['is_work_time'] == 1]

work_location = work_df.groupby('user_id')['geo_location'].agg([
    ('work_location',            most_frequent_geo),
    ('work_location_confidence', geo_confidence),
    ('work_txn_count',           'count'),
]).reset_index()

# Merge home and work to detect remote workers
work_location = work_location.merge(
    home_location[['user_id', 'home_location']],
    on='user_id',
    how='left'
)

# If work location == home location → likely remote worker
work_location['is_remote_worker'] = (
    work_location['work_location'] == work_location['home_location']
).astype(int)

# Flag reliable work locations
work_location['work_location_reliable'] = (
    (work_location['work_location_confidence'] >= 0.4) &
    (work_location['work_txn_count'] >= 3)
).astype(int)

print("\n✅ Work location computed.")
print(f"   Likely remote workers   : {work_location['is_remote_worker'].sum():,} users")
print(f"   Reliable work locations : {work_location['work_location_reliable'].sum():,} users")


# =============================================================================
# STEP 7 — MERGE ALL INTO USER PROFILE
# =============================================================================

user_profile = (
    age_features[['user_id', 'age_proxy_score', 'age_group',
                   'pct_young_mcc', 'pct_mature_mcc', 'pct_mobile']]
    .merge(
        income_features[['user_id', 'income_score', 'income_tier',
                          'avg_transaction', 'total_spend',
                          'pct_high_income_mcc', 'pct_budget_mcc']],
        on='user_id', how='left'
    )
    .merge(
        edu_features[['user_id', 'education_score', 'education_level',
                       'pct_edu_mcc', 'category_diversity']],
        on='user_id', how='left'
    )
    .merge(
        home_location[['user_id', 'home_location', 'home_location_confidence',
                        'home_location_reliable']],
        on='user_id', how='left'
    )
    .merge(
        work_location[['user_id', 'work_location', 'work_location_confidence',
                        'is_remote_worker', 'work_location_reliable']],
        on='user_id', how='left'
    )
)

print("\n" + "="*60)
print("✅ USER PROFILE COMPLETE")
print("="*60)
print(f"Shape: {user_profile.shape}")
print("\nSample:")
print(user_profile.head(5).to_string())

print("\n\nDemographic Summary:")
print("\nAge Groups:\n",       user_profile['age_group'].value_counts())
print("\nIncome Tiers:\n",     user_profile['income_tier'].value_counts())
print("\nEducation Levels:\n", user_profile['education_level'].value_counts())
print("\nRemote Workers:\n",   user_profile['is_remote_worker'].value_counts())


# =============================================================================
# STEP 8 — SAVE OUTPUT
# =============================================================================

user_profile.to_csv('user_demographic_profile.csv', index=False)
print("\n✅ Saved → user_demographic_profile.csv")


# =============================================================================
# CONFIDENCE SUMMARY TABLE
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║              FEATURE CONFIDENCE GUIDE                       ║
╠══════════════════════════════════╦═════════════╦════════════╣
║ Feature                          ║ Confidence  ║ Key Signal ║
╠══════════════════════════════════╬═════════════╬════════════╣
║ Income Tier                      ║ MEDIUM-HIGH ║ spend+MCC  ║
║ Home Location                    ║ MEDIUM-HIGH ║ geo+time   ║
║ Work Location                    ║ MEDIUM      ║ geo+time   ║
║ Age Group                        ║ MEDIUM      ║ MCC+device ║
║ Education Level                  ║ LOW         ║ MCC+merch  ║
╚══════════════════════════════════╩═════════════╩════════════╝

⚠️  All features are PROXIES — probabilistic estimates, not ground truth.
   Always validate against any available labeled data before modeling.
""")