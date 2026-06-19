#
# #
# #
# python -m scripts.seed_kb

from dotenv import load_dotenv
load_dotenv()

from rag.indexer import indexar_knowledge_base

def main():
    resultados = indexar_knowledge_base("data/knowledge_base/")
    for archivo, chunks in resultados.items():
        print(f"  {archivo}: {chunks} chunks")

main()