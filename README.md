# Wikipedia Title-Document Matching with BERT

This project utilizes the cl-tohoku BERT-base Japanese model to match Wikipedia titles with their corresponding document bodies. It predicts the most relevant document for each title from a set of candidates.

## Data Format

The input data should be in JSON format:
- `train_q.json`: Contains identified title-document pairs along with candidates for training.
- `test_q.json`: Contains titles with candidate documents for testing.
- `exam_data1.json`: Contains title and document IDs with their corresponding Japanese text.

## Usage

To run the project, follow these steps:

1. Clone the repository to your local machine.

2. Prepare your data and place it in the root directory of the project.

3. Load the BERT model and tokenizer:

    ```python
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
    ```

4. Train the model with the provided data:

    ```bash
    python train.py
    ```

5. After training, evaluate the model on the test set:

    ```bash
    python evaluate.py
    ```

6. Check the `suggestion.json` for the model's predictions on the test set.

## File Structure

- `README.md`: This file.
- `train.py`: Script for training the model.
- `evaluate.py`: Script for evaluating the model.
- `data/`: Directory where `train_q.json`, `test_q.json`, and `exam_data1.json` should be placed.

## Contributing

Contributions to this project are welcome! Please fork the repository and create a pull request with your improvements.

## License

This project is open-source and available under the MIT License. See the LICENSE file
