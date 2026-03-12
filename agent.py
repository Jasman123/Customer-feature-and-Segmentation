from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dotenv import load_dotenv
import time

load_dotenv()


def create_chat():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

llm = create_chat()


class GenderPrediction(BaseModel):
    gender: Literal["male", "female", "neutral"] = Field(
        description="Predicted gender target for this MCC category"
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    MCC_Code_name: str = Field(description="Name of the MCC category based on the code")


parser = PydanticOutputParser(pydantic_object=GenderPrediction)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a retail commerce analyst with deep knowledge of 
Merchant Category Codes (MCC).

Given only an MCC code, use your knowledge of what that category represents
to predict whether it primarily targets male or female customers, or is gender-neutral.

Rules:
- Return 'female' if the category skews toward female shoppers (e.g. 5621 = Women's Clothing)
- Return 'male' if the category skews toward male shoppers (e.g. 5533 = Auto Parts Stores)
- Return 'neutral' if the category serves both genders equally (e.g. 5411 = Grocery Stores)
- Confidence should reflect how strongly the MCC signals a gender

{format_instructions}"""),
    ("human", "MCC Code: {mcc_code}")
])

chain = prompt | llm | parser


def predict_gender(mcc_code: str) -> dict:
    """Predict gender target for a given MCC code."""
    try:
        result = chain.invoke({
            "mcc_code": mcc_code,
            "format_instructions": parser.get_format_instructions()
        })
        return {
            "mcc_code": mcc_code,
            "MCC_Code_name": result.MCC_Code_name,
            "gender": result.gender,
            "confidence": result.confidence,
        }
    except Exception as e:
        print(f"Error predicting for MCC '{mcc_code}': {e}")
        return {
            "mcc_code": mcc_code,
            "MCC_Code_name": "Unknown Category",
            "gender": "neutral",
            "confidence": 0.0,
        }


def predict_gender_batched(
    mcc_list: list,
    batch_size: int = 100,
    delay_seconds: float = 1.0,
    save_path: str = "mcc_gender_predictions.csv"
) -> pd.DataFrame:
    """
    Run gender prediction in batches with progress tracking and auto-save.

    Args:
        mcc_list      : Full list of MCC codes
        batch_size    : Number of MCCs processed per batch
        delay_seconds : Pause between batches to avoid rate limits
        save_path     : Output CSV path (saved after every batch)
    """
    total = len(mcc_list)
    total_batches = (total + batch_size - 1) // batch_size
    all_results = []

    print(f"📦 Total MCCs   : {total}")
    print(f"📏 Batch size   : {batch_size}")
    print(f"🔢 Total batches: {total_batches}")
    print(f"💾 Saving to    : {save_path}\n")

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, total)
        batch = mcc_list[start:end]

        print(f"🚀 Batch {batch_num + 1}/{total_batches} — items [{start + 1}–{end}]")

        for mcc in batch:
            result = predict_gender(str(mcc))
            all_results.append(result)
            print(f"   ✅ MCC {mcc} → {result['MCC_Code_name']} | {result['gender']} ({result['confidence']:.2f})")

        # Save checkpoint after every batch
        pd.DataFrame(all_results).to_csv(save_path, index=False)
        print(f"   💾 Checkpoint saved ({len(all_results)}/{total})\n")

        # Delay between batches (skip after last batch)
        if batch_num < total_batches - 1:
            time.sleep(delay_seconds)

    final_df = pd.DataFrame(all_results)
    print(f"✅ Done! {len(final_df)} predictions saved to '{save_path}'")
    return final_df


# ── Main ──────────────────────────────────────────────────────────────────────
mcc_list = pd.read_csv("list_mercant_id.csv")["merchant_id"].dropna().unique().tolist()
print(f"📊 Data shape: {len(mcc_list)} unique MCC codes\n")

results_df = predict_gender_batched(
    mcc_list=mcc_list,
    batch_size=100,     # 6489 codes → 65 batches
    delay_seconds=1.0,  # increase to 2.0+ if hitting rate limits
    save_path="mcc_gender_predictions.csv"
)

print(results_df.head(10))