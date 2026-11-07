"""Inspect installed DSV4.1 source for the graph-admission fix's invariants."""
import ast
from pathlib import Path
import sys

REQUIRED = {
    'compressor.py': ['CompressorMetadataBuilder'],
    'nvidia/flashinfer_sparse.py': ['DeepseekV4FlashInferSparseMLAMetadataBuilder', 'DeepseekSparseSWAFlashInferMetadataBuilder'],
    'nvidia/flashmla.py': ['DeepseekSparseSWAFlashMLAMetadataBuilder'],
    'sparse_mla.py': ['DeepseekV4FlashMLAMetadataBuilder'],
}

def check(root):
    results = {}
    for filename, names in REQUIRED.items():
        tree = ast.parse((Path(root)/filename).read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for name in names:
            node = classes.get(name)
            if node is None: raise ValueError(f'{filename}: missing {name}')
            text = ast.unparse(node)
            if 'AttentionCGSupport.UNIFORM_BATCH' not in text or 'def get_varlen_cudagraph_max_query_len' not in text or 'return max_decode_query_len(' not in text:
                raise ValueError(f'{filename}: graph fix invariant absent in {name}')
            results[name] = 'invariants-present'
    return results

if __name__ == '__main__':
    print(check(sys.argv[1]))
