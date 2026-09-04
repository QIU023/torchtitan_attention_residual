import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'torchtitan/trainer.py'; s = p.read_text()
old = '                            _f.write(f"{_n} {_g.float().norm().item():.10e} {tuple(_g.shape)}\\n")\n'
assert s.count(old) == 1, s.count(old)
new = '''                            import hashlib as _hl

                            _h = _hl.sha1(_g.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
                            _f.write(f"{_n} {_g.float().norm().item():.10e} {tuple(_g.shape)} {_g.dtype} {_h}\\n")
'''
p.write_text(s.replace(old, new)); print('grad hash hack applied to', sys.argv[1])
