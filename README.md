# ARI5118 (Deep Learning for Computer Vision) Assignment
The purpose of this GitHub repository is to host any code required for the ARI5118 Deep Learning for Computer Vision study-unit assignment.

Contact: [Michael Vella](michael.vella.20@um.edu.mt).

## Project notes:

- **Project Topic**: The project focuses on explaining, demonstrating and showcasing three foundational deep learning areas, which are Pooling, Normalisation and Activation Functions.
- **Python version used**: 3.12.3.
- **Packages used**: Refer to packages inside `requirements.txt`.

## Replication of virtual environment (.venv)

Assuming that Python is already pre-installed on the host machine and the repository is already cloned locally.

1. Run `python -m venv .venv` to create the Python virtual environment. `python` here refers to the alias of the Python executable path and depends on the alias used on the host machine (full Python path can also be used). Running this command will create a Python virtual environment depending on the base Python version being used to create the environment.
2. Activate virtual environment by running `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux).
3. Upgrade `pip` (Python's package manager) by running `pip install --upgrade pip`.
4. Run `pip install -r requirements.txt` to download any packages required for this project.

Your local setup is finished and you can now run the code.

## Project Directory

| File / Directory | Description |
|---|---|
| **further_reading/** | Academic papers that can be used as additonal reading/supporting material to the study notes. |
| **simulator/** | Streamlit based simulator highlighting the main concepts in pooling, normalisation and activation functions. The app is deployed online and can be accessed from [here](https://mvella-uom-ari5118-deep-learning.streamlit.app/). App can also be cloned and hosted locally - refer to `simulator/README.md` for more information. |
| **ai_journal.pdf** | Report showcasing how AI was used in this project. |
| **quiz_with_rationale.pdf** | PDF explaining the rational behind the correct/incorrect quiz answers. The actual quiz (Google Forms) can be accessed from [here](https://forms.gle/KorA8o9L2ATYVAqC9). |
| **requirements.txt** | List of packages required for this project (excluding packages required for the simulator as the simulator has a separate `requirements.txt` file). |
| **slides.pdf** | Slide deck used for the presentation. |
| **study_notes.pdf** | Full study notes on Pooling, Normalisation and Activation Functions. Also Includes the algorithms, worked examples, quick reference material and key papers. |
| **walkthrough.ipynb** | Annotated code walkthrough highlighting the main concepts in Pooling, Normalisation and Activation Functions. |
