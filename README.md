
# EcoSort: Intelligent Waste Management Assistant

**Making waste segregation smarter with AI.**

EcoSort is an AI-driven waste management assistant designed to help residents identify waste materials, classify them into appropriate categories, and access reliable disposal guidance based on local recycling policies.

The system integrates **computer vision, natural language processing (NLP), and retrieval-augmented generation (RAG)** into a unified pipeline. Residents can submit an image of a waste item or describe it in text, receive a predicted waste category, and obtain disposal recommendations grounded in Metro City's waste management policies.

Developed in partnership with Metro City's waste management department.

---

## Project Objectives

Incorrect waste segregation can contaminate recyclable materials, increase landfill waste, and make recycling less efficient. EcoSort aims to address these challenges by making waste classification and disposal guidance more accessible.

The project focuses on three objectives:

- **Automate waste classification:** Use AI to identify waste categories from images and text descriptions.
- **Provide locally relevant guidance:** Retrieve applicable recycling and disposal rules from official policy documents.
- **Improve waste disposal decisions:** Translate classification results and policy information into clear, actionable instructions.

##  Core Capabilities

| Component | Technology | Purpose |
|---|---|---|
| Image Classification | Convolutional Neural Network (CNN) | Predicts waste categories from uploaded images. |
| Text Classification | NLP models (RNN / Transformer) | Classifies waste based on written descriptions. |
| Policy Retrieval | Sentence embeddings and semantic similarity | Retrieves relevant passages from waste management policies. |
| Response Generation | Retrieval-Augmented Generation (RAG) | Generates disposal instructions grounded in retrieved policy information. |

### 1. Computer Vision

EcoSort uses a CNN-based image classification pipeline to recognize waste materials from user-uploaded photographs.

The image classification component is designed to distinguish between nine categories:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

The predicted category provides the first step toward identifying appropriate disposal options.

### 2. Natural Language Processing

Not all waste items can be identified easily from photographs. Users may instead describe an item using details about its material, composition, or condition.

EcoSort incorporates an NLP classification component to interpret these descriptions and predict the corresponding waste category.

The intended pipeline is:

**Text description → Text preprocessing → NLP classification → Predicted waste category**

RNN and Transformer architectures are relevant approaches for this component, with the final choice depending on the model implemented and its evaluation results.

### 3.  Retrieval-Augmented Generation (RAG)

A waste category alone may not provide enough information to determine the correct disposal method. Local policies can specify different handling requirements depending on the material and its condition.

EcoSort addresses this challenge through RAG, combining semantic document retrieval with generative AI.

The process follows these steps:

1. Receive the waste category and the user's query.
2. Convert the query into a vector representation using a sentence-embedding model.
3. Retrieve semantically relevant passages from the policy documents.
4. Provide the retrieved context to a generative model.
5. Generate disposal instructions grounded in the retrieved policy information.

This approach is intended to make recommendations more relevant to local requirements rather than relying solely on general-purpose model knowledge.

---

## Datasets

EcoSort integrates three data sources to support image classification, text classification, and policy retrieval.

| Dataset | Size | Description | Primary Use |
|---|---:|---|---|
| **RealWaste** | 4,752 images | Images across nine waste categories. | Training and evaluating the image classification model. |
| **Waste Descriptions** | 5,000 records | Text descriptions, waste categories, and material composition. | Developing and evaluating the text classification model. |
| **Metro City Policies** | 14 documents | Waste handling, recycling, and disposal guidance. | Building the knowledge base for RAG. |

### Dataset Organization

The project follows a structure that separates the image dataset, structured text data, policy documents, and source code.

```text
data/
├── waste_descriptions.csv
└── waste_policy_documents.json

RealWaste/
├── Cardboard/
├── Food Organics/
├── Glass/
├── Metal/
├── Miscellaneous Trash/
├── Paper/
├── Plastic/
├── Textile Trash/
└── Vegetation/
```

**Data quality considerations**

- Verify image labels and class distributions before training.
- Inspect the text dataset for missing values, duplicate records, inconsistent labels, and class imbalance.
- Use appropriate training, validation, and test splits when evaluating classification models.
- Ensure policy documents reflect the applicable local rules and retain their source information and update dates.

Dataset provenance, usage permissions, and any applicable data privacy requirements should be documented before distributing the complete dataset.

---

## System Architecture

EcoSort combines two classification pathways with a shared policy retrieval and response-generation stage.

```text
                    USER INPUT
                        |
             +----------+----------+
             |                     |
        Image Input           Text Input
             |                     |
       Image Preprocessing    Text Preprocessing
             |                     |
          CNN Model             NLP Model
             |                     |
             +----------+----------+
                        |
                Predicted Category
                        |
              User Query / Context
                        |
              Sentence Embeddings
                        |
              Semantic Retrieval
                        |
             Metro City Policy Data
                        |
                Relevant Passages
                        |
              Generative AI Model
                        |
               Disposal Guidance
```

### End-to-End Workflow

**Input:** A photograph of a waste item or a written description.

**Classification:** The appropriate model predicts a waste category.

**Retrieval:** The system searches the policy knowledge base for relevant disposal rules.

**Generation:** A generative model uses the retrieved passages to formulate disposal guidance.

**Output:** The user receives a predicted category and relevant disposal instructions.

The classification and retrieval components should be evaluated independently as well as together, since an incorrect classification can affect the relevance of the final recommendation.

---

## Project Structure

```text
Integrated_Waste_Management_Assistant/
│
├── data/
│   ├── waste_descriptions.csv
│   └── waste_policy_documents.json
│
├── RealWaste/
│   └── [Nine waste-category folders]
│
├── ecosort.py
├── waste_management.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

| File or Directory | Description |
|---|---|
| `data/` | Contains structured text data and waste policy documents. |
| `RealWaste/` | Contains the image dataset organized by waste category. |
| `ecosort.py` | Consolidated script for the implemented EcoSort pipeline. |
| `waste_management.ipynb` | Notebook for exploration, experimentation, model development, and analysis. |
| `requirements.txt` | Lists the Python dependencies required by the project. |
| `LICENSE` | Specifies the project's licensing terms. |
| `README.md` | Documents the project, architecture, and usage. |

---

## Installation and Setup

### Prerequisites

Before getting started, ensure you have:

- Python installed in your environment.
- Git installed on your machine.
- Access to the required datasets.
- Jupyter Notebook or JupyterLab for interactive development.

### 1. Clone the Repository

```bash
git clone https://github.com/ewangila/Integrated_Waste_Management_Assistant.git
```

Navigate into the project directory:

```bash
cd Integrated_Waste_Management_Assistant
```

### 2. Create a Virtual Environment

Creating an isolated environment helps prevent dependency conflicts.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Deep learning frameworks and model dependencies should match the implementations used in the project. TensorFlow and PyTorch do not necessarily both need to be installed; retain the dependencies required by the actual pipeline.

---

## Usage

EcoSort can be explored through the development notebook or executed through the consolidated Python script.

### Option 1: Jupyter Notebook

The notebook is recommended for interactive exploration, experimentation, and reviewing the development process.

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
waste_management.ipynb
```

Run the cells in order, following any dataset preparation and model configuration instructions documented in the notebook.

### Option 2: Python Script

Run the consolidated pipeline from the project root:

```bash
python ecosort.py
```

The script's available inputs, outputs, and execution behavior depend on the functionality implemented in `ecosort.py`.

Ensure that the required datasets and any model files are available at the paths expected by the script.

---

## Model Evaluation

Reliable waste classification and policy-grounded recommendations require more than a successful model run. Each component should be evaluated against appropriate performance criteria.

| Component | Recommended Evaluation Metrics | What to Assess |
|---|---|---|
| Image Classification | Accuracy, precision, recall, F1-score, confusion matrix | Classification performance across all nine waste categories. |
| Text Classification | Accuracy, precision, recall, F1-score, confusion matrix | Correct classification of written waste descriptions. |
| Policy Retrieval | Precision@k, Recall@k, human relevance judgments | Whether the retrieved passages contain the information needed to answer a query. |
| Response Generation | Groundedness, factual correctness, relevance, policy consistency | Whether generated instructions are supported by the retrieved documents. |
| End-to-End Pipeline | Classification correctness, policy relevance, answer correctness | Whether the complete workflow produces appropriate disposal guidance. |

### Evaluation Considerations

- Use held-out test data to estimate generalization performance.
- Review precision and recall for individual classes, especially where misclassification could lead to incorrect disposal.
- Check for data leakage between training and test sets.
- Evaluate whether retrieved passages support the generated recommendations.
- Test the system with ambiguous descriptions, unfamiliar objects, and waste items that do not fit neatly into one category.
- Document actual evaluation results before making claims about model accuracy or system reliability.

**Model performance results are intentionally not reported here.** They should be added after the implemented models have been evaluated.

---

## Responsible Use and Limitations

EcoSort is designed to assist residents with waste sorting. Its recommendations should not be treated as a substitute for authoritative waste management guidance.

Important considerations include:

- **Classification uncertainty:** Similar-looking materials may be difficult to distinguish from images alone.
- **Ambiguous descriptions:** Short or incomplete descriptions may not contain enough information for reliable classification.
- **Policy currency:** Disposal rules can change, so the policy knowledge base must be maintained.
- **Retrieval quality:** Relevant instructions may not be returned when a query uses unfamiliar terminology or when policy documents are incomplete.
- **Generation accuracy:** A generative model can misinterpret retrieved information or introduce unsupported claims. Outputs should be checked against the source documents.
- **Out-of-scope materials:** Hazardous, medical, electronic, or other specially regulated waste may require dedicated handling guidance beyond the nine classification categories.

Where the system cannot produce sufficiently reliable guidance, it should avoid presenting uncertain recommendations as definitive instructions and direct users to the relevant local authority.

---

## Potential Future Enhancements

Potential extensions to the project include:

- **Interactive web application:** Build a user-friendly interface for image uploads, text input, and disposal recommendations.
- **Confidence-aware predictions:** Display prediction confidence and provide a fallback when classification is uncertain.
- **Expanded waste categories:** Support additional materials and special waste-handling requirements.
- **Policy monitoring:** Track policy source dates and provide a process for updating the knowledge base when official rules change.
- **Multilingual support:** Enable residents to describe waste and access guidance in additional languages.
- **Performance monitoring:** Track model performance, retrieval relevance, and recurring classification errors.
- **User feedback:** Allow residents to flag incorrect classifications or disposal instructions for review.

These are proposed enhancements rather than claims about currently implemented features.

---

## 🧰 Technology Stack

The project uses Python-based tools for data processing, machine learning, deep learning, and semantic retrieval.

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | pandas, NumPy |
| Image Processing | Pillow |
| Deep Learning | TensorFlow or PyTorch |
| NLP | Transformers |
| Text Embeddings | Sentence Transformers |
| Machine Learning | scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebook |

The exact dependency list is maintained in `requirements.txt`.

---

## 📄 License

This project is released under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full terms.

The project license does not automatically grant permission to redistribute third-party datasets, pretrained models, or official policy documents. Their respective licenses and usage conditions should be checked separately.

---

## 👤 Author

**Eugin Wangila Wangwe**

GitHub: [@ewangila](https://github.com/ewangila)

Project Repository: [Integrated Waste Management Assistant](https://github.com/ewangila/Integrated_Waste_Management_Assistant)

---

## 🌍 Project Vision

EcoSort explores how artificial intelligence can support more informed waste disposal decisions by combining visual recognition, text understanding, and locally relevant policy information.

By bringing these capabilities together, the project aims to make waste segregation guidance easier to access, understand, and apply in everyday life.

**Smarter sorting. Better disposal decisions. A cleaner environment.**
````
