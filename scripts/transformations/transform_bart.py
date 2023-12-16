from src.processing.bart_processing import RelationConverter

if __name__ == "__main__":
    
    # input_path = '/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-rebalPairs-rwrtKeys-instrC-mxTrnCp3-skpTps'
    input_path = '/home/murilo/RelNetCare/data/processed/dialog-re-37cls-with-no-relation-llama-48cls-clsTskOnl-instrC-shfflDt-skpTps-37clsWthNRltn'
    cls_only = 'clsTskOnl' in input_path
    converter = RelationConverter(input_path=input_path, cls_only=cls_only)
    converter.process_llama_json_to_bart_sentence()
    # converter.test_conversion()
