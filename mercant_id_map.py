MCC_RANGES = [
    (1,    1499, "Agricultural Services"),
    (1500, 2999, "Contracted Services"),
    (3000, 3999, "Travel & Airlines"),
    (4000, 4799, "Transportation Services"),
    (4800, 4999, "Utility Services"),
    (5000, 5599, "Retail Outlet Services"),
    (5600, 5699, "Clothing Stores"),
    (5700, 7299, "Miscellaneous Stores"),
    (7300, 7999, "Business Services"),
    (8000, 8999, "Professional Services"),
    (9000, 9999, "Government Services"),
]

def get_merchant_category(merchant_category_id):
    for lower, upper, category in MCC_RANGES:
        if lower <= merchant_category_id <= upper:
            return category
    return "Unknown Category"