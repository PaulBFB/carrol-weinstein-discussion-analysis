import loguru
from tqdm import tqdm
import ollama
import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


logger = loguru.logger


class Concept(BaseModel):
    concept_name: str = Field(
        description="the name of concept from physics or mathematics mentioned in the text"
    )
    explanation_of_concept: str = Field(
        description="a brief explanation of what the concept is"
    )
    education_level: str = Field(
        description="the expected level of education at which students are exposed to the concepts for the first time"
    )


class ResponseModel(BaseModel):
    response: list[Concept]


def get_structured_response(
    text_to_analyze: str, pydantic_response_model: BaseModel = ResponseModel
):
    """Get structured response using Pydantic model for validation"""

    # Generate JSON schema from Pydantic model
    schema_json = pydantic_response_model.model_json_schema()
    schema_str = json.dumps(schema_json, indent=2)

    # Simplified schema description instead of full JSON schema
    system_prompt = """You are a helpful assistant that responds only with valid JSON.

    Analyze the provided transcript for physics or mathematics concepts.
    Only search for concepts related to physics or mathematics, nothing else.
    Respond with this exact JSON structure:

    {
      "response": [
        {
          "concept_name": "name of the concept",
          "explanation_of_concept": "brief explanation",
          "education_level": "education level (e.g., High School, Undergraduate, Graduate)"
        }
      ]
    }

    If no physics/math concepts are found, return: {"response": []}

    Respond with ONLY the JSON, no additional text or explanations."""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_to_analyze},
        ],
        options={"temperature": 0.0},
    )

    # Parse and validate with Pydantic
    json_data = json.loads(response["message"]["content"])

    try:
        result = pydantic_response_model.model_validate(json_data)
    except ValidationError:
        result = pydantic_response_model.model_validate(json_data["response"])

    return result


def add_concept_labels_to_dataframe(
    data: pd.DataFrame, text_column_label: str = "text"
) -> pd.DataFrame:
    assert "text" in data.columns, f"missing text column - label {text_column_label}"

    topics = []

    logger.info(f"creating concepts for data - {data.shape}")
    with tqdm(total=data.shape[0]) as pbar:
        for _, row in tqdm(data.iterrows()):
            row_topics = get_structured_response(text_to_analyze=row[text_column_label])
            data = row_topics.model_dump()
            data.update(row.to_dict())
            topics.append(data)
            pbar.update(1)

    topics_df = pd.DataFrame(topics)
    topics_df = topics_df.explode("response").reset_index()

    # expand into columns
    columns_to_use: list = [c for c in topics_df.columns if c != "response"]

    topics_df = pd.concat(
        [topics_df[columns_to_use], topics_df["response"].apply(pd.Series)],
        axis=1,
    )
    topics_df = topics_df.drop(columns=[0])

    return topics_df


if __name__ == "__main__":
    from pathlib import Path

    jack_data_path = Path.cwd() / "data" / "science man vs noted idiots.txt"
    with open(jack_data_path, mode="r") as file:
        jack_df = pd.read_csv(file, sep="\t")

    test_text: str = "what is your main bone of contention with Eric Weinstein uh you should probably ask Eric that sean has been nothing but civil he's also extremely nasty so let me say a bunch of things to Dr carol dr carol I'd like to hear your explanation for three generations of flavored chyro firmians with the observed quantum numbers under the group SU3 plus SU2 SU1 i have read Eric's paper here it is i actually have it here it's worse than you would think how dare you how dare I read your paper i highly advise you to spend more time in your physics department and less time on YouTube so we're not allowed to think about Eric's theory and write a follow-up paper about it and although you're very much Sean I have my disagreements with string theorists my agreements with them but I respect it and I think that they're trying their best the first rule of physics fight club is don't talk about the problems with physics fight club i have found myself in the awkward and unenviable position of defending the establishment heterodoxy your intellectually insulting"

    test_concepts = get_structured_response(test_text)
    test_topics = add_concept_labels_to_dataframe(jack_df.head())

    print(test_concepts)
