"""Check actual tool/system request histories against the installed encoder."""
import argparse
import hashlib
import json
from pathlib import Path

import verify_coding_capture


def build(tokenizer, root, *, whole_panel=True):
    root=Path(root)
    if whole_panel:
        population=verify_coding_capture.verify(root)
        directories=[root/task['id'] for task in json.loads((root/'panel.json').read_text())]
    else:
        population=verify_coding_capture.conversation(root);directories=[root]
    rows=[]
    for directory in directories:
        conversation=json.loads((directory/'conversation.json').read_text())
        for turn in conversation['turns']:
            index=turn['turn'];request_path=directory/f'turn-{index}-request.json'
            raw_path=directory/f'turn-{index}-raw.json'
            body=json.loads(request_path.read_text())['body']
            _,_,usage=verify_coding_capture.response(json.loads(raw_path.read_text()))
            kwargs=body['chat_template_kwargs']
            rendered=tokenizer.apply_chat_template(body['messages'],tools=body.get('tools'),
                thinking=kwargs['thinking'],reasoning_effort=kwargs['reasoning_effort'],tokenize=False)
            if not isinstance(rendered,str):raise ValueError('installed history renderer did not return text')
            ids=tokenizer.encode(rendered,add_special_tokens=False)
            if not ids or any(type(n) is not int or n<0 for n in ids):raise ValueError('installed history token IDs invalid')
            count=len(ids);difference=usage['prompt_tokens']-count
            rows.append({'task_id':conversation['task_id'],'turn':index,
                'request_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),
                'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                'token_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                'rendered_tokens':count,'observed_prompt_tokens':usage['prompt_tokens'],'difference':difference,
                'tools':len(body.get('tools',[])),'messages':len(body['messages']),
                'output_room_tokens':1048576-count,'requested_output_cap':body['max_tokens'],
                'exact_match':difference==0 and count+body['max_tokens']<=1048576})
    if not rows:raise ValueError('coding history population is empty')
    return {'schema':'coding-installed-history/v1','status':'matched' if all(r['exact_match'] for r in rows) else 'mismatch',
            'whole_panel':whole_panel,'captured_tasks':len(directories),'requests':len(rows),'rows':rows,
            'capture_population':population,
            'scope':'actual captured request histories including tool schemas and mid-conversation system; exact encoder/usage comparison, not executable correctness'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model');parser.add_argument('capture');parser.add_argument('--single-task',action='store_true')
    args=parser.parse_args()
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    tokenizer=DeepseekV41Tokenizer.from_pretrained(args.model,local_files_only=True)
    result=build(tokenizer,args.capture,whole_panel=not args.single_task)
    source=Path(tokenizer.apply_chat_template.__code__.co_filename)
    result['installed_renderer']={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    result['checkpoint_tokenizer_sha256']=hashlib.sha256((Path(args.model)/'tokenizer.json').read_bytes()).hexdigest()
    print(json.dumps(result,sort_keys=True),flush=True)
    raise SystemExit(0 if result['status']=='matched' else 1)


if __name__=='__main__':main()
