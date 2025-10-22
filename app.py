import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the trained model and tokenizer for inference
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./results/checkpoint/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None
    ).to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Color mapping for different entity types
ENTITY_COLORS = {
    "PERSON": "#aa9cfc",
    "NORP": "#feca74",
    "FAC": "#9cc9cc",
    "ORG": "#7aecec",
    "GPE": "#ffa8a8",
    "LOC": "#ff9561",
    "PRODUCT": "#bfe1d9",
    "EVENT": "#ffeb80",
    "WORK_OF_ART": "#bfde9c",
    "LAW": "#f0d0ff",
    "LANGUAGE": "#b4a7d6",
    "DATE": "#bfe1d9",
    "TIME": "#bfe1d9",
    "PERCENT": "#e4e7d2",
    "MONEY": "#e4e7d2",
    "QUANTITY": "#e4e7d2",
    "ORDINAL": "#e4e7d2",
    "CARDINAL": "#e4e7d2"
}

# Inference function that returns structured entities
def ner_inference(text):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [model.config.id2label[pred] for pred in predictions[0]]

    # Reconstruct entities with their positions
    entities = []
    current_entity = None
    
    for idx, (token, label, offset) in enumerate(zip(tokens, predicted_labels, offset_mapping)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if label.startswith("B-"):
            # Start of a new entity
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                "text": text[offset[0]:offset[1]],
                "label": entity_type,
                "start": offset[0].item(),
                "end": offset[1].item()
            }
        elif label.startswith("I-") and current_entity:
            # Continue current entity
            current_entity["text"] = text[current_entity["start"]:offset[1]]
            current_entity["end"] = offset[1].item()
        elif label == "O":
            # No entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# Create displaCy-style HTML visualization
def render_ner_html(text, entities):
    """Generate HTML markup for NER visualization similar to displaCy"""
    
    html_parts = ['<div style="line-height: 2.5; direction: ltr; font-family: sans-serif; font-size: 16px;">']
    
    last_end = 0
    for entity in sorted(entities, key=lambda x: x["start"]):
        # Add text before entity
        if entity["start"] > last_end:
            html_parts.append(text[last_end:entity["start"]])
        
        # Add entity with styling
        color = ENTITY_COLORS.get(entity["label"], "#ddd")
        html_parts.append(
            f'<mark style="background: {color}; padding: 0.25em 0.4em; margin: 0 0.15em; '
            f'line-height: 1; border-radius: 0.35em; display: inline-block; vertical-align: middle;">'
            f'{entity["text"]}'
            f'<span style="font-size: 0.7em; font-weight: bold; line-height: 1; border-radius: 0.35em; '
            f'vertical-align: middle; margin-left: 0.4rem; background: rgba(0, 0, 0, 0.1); padding: 0.2em 0.4em;">'
            f'{entity["label"]}'
            f'</span>'
            f'</mark>'
        )
        last_end = entity["end"]
    
    # Add remaining text
    if last_end < len(text):
        html_parts.append(text[last_end:])
    
    html_parts.append('</div>')
    
    return ''.join(html_parts)

st.set_page_config(page_title="DistilBert NER App", layout="wide")

st.title("Named Entity Recognition App")

st.sidebar.title("About")
st.sidebar.info("This app uses the DistilBert model for Named Entity Recognition (NER) on user-provided text.")

# Add details about which spacy tag means what
st.sidebar.markdown("""
**Entity Labels:**
- `PERSON`: People, including fictional
- `NORP`: Nationalities or religious/political groups
- `FAC`: Buildings, airports, highways, bridges, etc.
- `ORG`: Companies, agencies, institutions, etc.
- `GPE`: Countries, cities, states
- `LOC`: Non-GPE locations, mountain ranges, bodies of water
- `PRODUCT`: Objects, vehicles, foods, etc. (not services)
- `EVENT`: Named hurricanes, battles, wars, sports events, etc.
- `WORK_OF_ART`: Titles of books, songs, etc.
- `LAW`: Named documents made into laws
- `LANGUAGE`: Any named language
- `DATE`: Absolute or relative dates or periods
- `TIME`: Times smaller than a day
- `PERCENT`: Percentage, including "%"-sign
- `MONEY`: Monetary values, including "$"-sign
- `QUANTITY`: Measurements, as of weight or distance
- `ORDINAL`: "first", "second", etc.
- `CARDINAL`: Numerals that do not fall under another type
""")

user_input = st.text_area("Enter text for NER", placeholder="Type your text here...", height=150)

if st.button("Analyze", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing text..."):
            entities = ner_inference(user_input)
        
        if entities:
            st.subheader("NER Results")
            # Display the displaCy-style visualization
            html_output = render_ner_html(user_input, entities)
            st.markdown(html_output, unsafe_allow_html=True)
            
        else:
            st.info("No entities found in the text.")
    else:
        st.warning("Please enter some text to analyze.")

