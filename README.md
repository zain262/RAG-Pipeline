# RAG-Pipeline

## Requirements
- Python 3.10+
- pip

Install dependencies:
pip install -r requirements.txt

Install Ollama
Download from: https://ollama.com
ollama pull llama3 or ollama pull mistral
ollama serve

Run st1:
 py -3.10 run_semeval_st1_rag.py  --data_path food_recall_incidents.csv  --run_splits validation   --max_docs 300   --top_k 5 --model mistral:7b 

Run st2:
py -3.10 run_semeval_st2_simple.py  --data_path food_recall_incidents.csv  --run_splits validation   --max_docs 300   --top_k 15   --limit_rows 200 --model llama3.1:8b

Flags:
--data_path:	Dataset location
--model:	LLM used
--api_base_url:	Ollama API endpoint
--top_k:	Number of retrieved docs
--max_docs:	Max indexed documents
--run_splits:	Dataset splits to run
--out_csv:	Output predictions file
--max_tokens:	LLM response length

python run_semeval_st2_simple.py \
  --data_path /path/to/data \
  --run_splits validation \
  --model llama3.1:8b \
  --api_base_url http://localhost:11434/v1 \
  --top_k 8 \
  --max_docs 3000 \
  --limit_rows 0 \
  --max_tokens 220 \
  --out_csv predictions_st2.csv

  

  The only required field is the data_path field, all others can be left blank and will be populated with the default value shown above.
