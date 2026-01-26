# RAG-Pipeline
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

  ex.
  python run_semeval_st2_simple.py --data_path food_recall_incidents.csv --run_splits validation --max_docs 3000

  The only required field is the data_path field, all others can be left blank and will be populated with the default value shown above.
