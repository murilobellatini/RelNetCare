from src.processing.bart_processing import RelationConverter

if __name__ == "__main__":
    input_path = '/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps'
    converter = RelationConverter(input_path=input_path)
    converter.process_llama_json_to_bart_sentence()
    # converter.test_conversion()
