#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ctypes, shutil, subprocess, tempfile, unittest, zlib, struct
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
FP=ctypes.POINTER(ctypes.c_float)

class LayerCfg(ctypes.Structure):
    _fields_=[('is_local',ctypes.c_int),('num_heads',ctypes.c_int),('num_kv_heads',ctypes.c_int),('head_dim',ctypes.c_int),('relative_extent',ctypes.c_int)]
class Config(ctypes.Structure):
    _fields_=[('n_layers',ctypes.c_int),('hidden',ctypes.c_int),('vocab',ctypes.c_int),('unpadded_vocab',ctypes.c_int),('max_context',ctypes.c_int),('global_heads',ctypes.c_int),('global_kv_heads',ctypes.c_int),('global_head_dim',ctypes.c_int),('local_heads',ctypes.c_int),('local_kv_heads',ctypes.c_int),('local_head_dim',ctypes.c_int),('sliding_window',ctypes.c_int),('d_rel',ctypes.c_int),('rel_extent',ctypes.c_int),('conv_kernel',ctypes.c_int),('dense_layers',ctypes.c_int),('dense_intermediate',ctypes.c_int),('moe_intermediate',ctypes.c_int),('n_routed_experts',ctypes.c_int),('top_k',ctypes.c_int),('n_shared_experts',ctypes.c_int),('rms_eps',ctypes.c_float),('route_scale',ctypes.c_float),('logits_width_multiplier',ctypes.c_float),('log_scaling_n_floor',ctypes.c_int),('log_scaling_alpha',ctypes.c_float),('layer',LayerCfg*128)]
class View(ctypes.Structure):
    _fields_=[('name',ctypes.c_char_p),('data',FP),('shape',ctypes.c_int*4),('ndim',ctypes.c_int)]
class Expert(ctypes.Structure):
    _fields_=[('gate',FP),('up',FP),('down',FP)]
class StageTensor(ctypes.Structure):
    _fields_=[('fd',ctypes.c_int),('payload_off',ctypes.c_uint64),('payload_bytes',ctypes.c_uint64),('stored_bytes',ctypes.c_uint64),('payload_crc32',ctypes.c_uint32),('shape',ctypes.c_uint32*4),('rows',ctypes.c_uint32),('cols',ctypes.c_uint32),('ndim',ctypes.c_int),('dtype',ctypes.c_int),('io',ctypes.POINTER(ctypes.c_ubyte)),('io_bytes',ctypes.c_size_t)]
class StageBank(ctypes.Structure):
    _fields_=[('fd',ctypes.c_int),('layer',ctypes.c_int),('experts',ctypes.c_int),('hidden',ctypes.c_int),('intermediate',ctypes.c_int),('record_bytes',ctypes.c_uint64),('verify_crc',ctypes.c_int),('gate',FP),('up',FP),('down',FP),('raw',ctypes.POINTER(ctypes.c_ubyte)),('matrix_floats',ctypes.c_size_t),('raw_bytes',ctypes.c_size_t),('owns_workspace',ctypes.c_int)]

def arr(t):
    a=(ctypes.c_float*t.numel())(*t.flatten().tolist()); return a

def bf16_bytes(t):
    return t.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()

class TestBindAndStage(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    cc=shutil.which('cc')
    if not cc: raise unittest.SkipTest('cc unavailable')
    cls.td=tempfile.TemporaryDirectory(); so=Path(cls.td.name)/'lib.so'
    subprocess.run([cc,'-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',f'-I{REPO/"src"}',str(REPO/'src/inkling_bind.c'),str(REPO/'src/inkling_stage_reader.c'),'-o',str(so)],check=True,capture_output=True)
    cls.lib=ctypes.CDLL(str(so)); cls.lib.waste_inkling_bind_weights.argtypes=[ctypes.c_void_p,ctypes.POINTER(Config),ctypes.POINTER(View),ctypes.c_size_t];cls.lib.waste_inkling_bind_weights.restype=ctypes.c_int
    cls.lib.waste_inkling_stage_tensor_open.argtypes=[ctypes.POINTER(StageTensor),ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int];cls.lib.waste_inkling_stage_tensor_open.restype=ctypes.c_int
    cls.lib.waste_inkling_stage_tensor_row.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,FP];cls.lib.waste_inkling_stage_tensor_row.restype=ctypes.c_int
    cls.lib.waste_inkling_stage_tensor_close.argtypes=[ctypes.POINTER(StageTensor)]
    cls.lib.waste_inkling_stage_bank_open.argtypes=[ctypes.POINTER(StageBank),ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_int];cls.lib.waste_inkling_stage_bank_open.restype=ctypes.c_int
    cls.lib.waste_inkling_stage_expert_get.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.POINTER(Expert)];cls.lib.waste_inkling_stage_expert_get.restype=ctypes.c_int
    cls.lib.waste_inkling_stage_bank_close.argtypes=[ctypes.POINTER(StageBank)]
 @classmethod
 def tearDownClass(cls): cls.td.cleanup()
 def cfg(self):
    c=Config();c.n_layers=1;c.hidden=4;c.vocab=7;c.unpadded_vocab=6;c.max_context=8;c.dense_layers=1;c.dense_intermediate=5;c.moe_intermediate=3;c.n_routed_experts=2;c.top_k=1;c.n_shared_experts=1;c.d_rel=2;c.conv_kernel=3;c.layer[0]=LayerCfg(1,2,1,2,4);return c
 def make_views(self,c):
    specs={'inkling.embed':(c.vocab,c.hidden),'inkling.embed_norm':(c.hidden,),'inkling.final_norm':(c.hidden,),'inkling.unembed':(c.unpadded_vocab,c.hidden),
    'inkling.layer.0.input_norm':(c.hidden,),'inkling.layer.0.post_attention_norm':(c.hidden,),'inkling.layer.0.q':(4,c.hidden),'inkling.layer.0.k':(2,c.hidden),'inkling.layer.0.v':(2,c.hidden),'inkling.layer.0.r':(4,c.hidden),'inkling.layer.0.o':(c.hidden,4),'inkling.layer.0.q_norm':(2,),'inkling.layer.0.k_norm':(2,),'inkling.layer.0.rel_proj':(2,4),'inkling.layer.0.k_sconv':(2,3),'inkling.layer.0.v_sconv':(2,3),'inkling.layer.0.attn_sconv':(4,3),'inkling.layer.0.mlp_sconv':(4,3),'inkling.layer.0.mlp.gate':(5,4),'inkling.layer.0.mlp.up':(5,4),'inkling.layer.0.mlp.down':(4,5),'inkling.layer.0.mlp.global_scale':(1,)}
    keep=[]; views=[]
    for name,shape in specs.items():
      a=arr(torch.randn(*shape));keep.append(a);s=(ctypes.c_int*4)(*(list(shape)+[0]*(4-len(shape))));views.append(View(name.encode(),a,s,len(shape)))
    return (View*len(views))(*views),keep
 def test_registry_accepts_exact_and_rejects_shape(self):
    c=self.cfg();v,k=self.make_views(c);out=(ctypes.c_byte*65536)();self.assertEqual(self.lib.waste_inkling_bind_weights(out,ctypes.byref(c),v,len(v)),0);v[0].shape[1]+=1;self.assertEqual(self.lib.waste_inkling_bind_weights(out,ctypes.byref(c),v,len(v)),-1);self.assertTrue(k)
 def test_bf16_tensor_row_and_crc(self):
    name='inkling.embed';x=torch.tensor([[1.25,-2.5],[3.0,4.5]],dtype=torch.float32);p=bf16_bytes(x);shape=(2,2,0,0);hdr=struct.pack('<4sHBBHH4IQII20s',b'IKTN',1,1,2,0,0,*shape,len(p),zlib.crc32(p)&0xffffffff,zlib.crc32(name.encode())&0xffffffff,b'\0'*20)
    path=Path(self.td.name)/'t.stage';path.write_bytes(hdr+p+b'\0'*(4096-64-len(p)));t=StageTensor();sh=(ctypes.c_int*2)(2,2);self.assertEqual(self.lib.waste_inkling_stage_tensor_open(ctypes.byref(t),str(path).encode(),name.encode(),sh,2,1),0);o=(ctypes.c_float*2)();self.assertEqual(self.lib.waste_inkling_stage_tensor_row(ctypes.byref(t),1,2,o),0);torch.testing.assert_close(torch.tensor(list(o)),x[1],rtol=0,atol=.02);self.lib.waste_inkling_stage_tensor_close(ctypes.byref(t));raw=bytearray(path.read_bytes());raw[65]^=1;path.write_bytes(raw);self.assertEqual(self.lib.waste_inkling_stage_tensor_open(ctypes.byref(t),str(path).encode(),name.encode(),sh,2,1),-1)
 def test_bf16_expert_record_identity_and_crc(self):
    h,i=3,2;g=torch.arange(i*h,dtype=torch.float32).reshape(i,h)/4;u=g+1;d=torch.arange(h*i,dtype=torch.float32).reshape(h,i)/7;payload=bf16_bytes(g)+bf16_bytes(u)+bf16_bytes(d);rec=4096;hdr=struct.pack('<IHHHBBIIQQQQI8x',0x46424B49,1,2,0,1,0,h,i,64,64+len(bf16_bytes(g)),64+len(bf16_bytes(g))+len(bf16_bytes(u)),len(payload),zlib.crc32(payload)&0xffffffff);path=Path(self.td.name)/'bank';path.write_bytes(hdr+payload+b'\0'*(rec-64-len(payload)));b=StageBank();self.assertEqual(self.lib.waste_inkling_stage_bank_open(ctypes.byref(b),str(path).encode(),2,1,h,i,rec,1),0);e=Expert();self.assertEqual(self.lib.waste_inkling_stage_expert_get(ctypes.byref(b),2,0,ctypes.byref(e)),0);got=torch.tensor([e.gate[j] for j in range(i*h)]).reshape(i,h);torch.testing.assert_close(got,g,rtol=0,atol=.02);self.assertEqual(self.lib.waste_inkling_stage_expert_get(ctypes.byref(b),3,0,ctypes.byref(e)),-1);self.lib.waste_inkling_stage_bank_close(ctypes.byref(b))
if __name__=='__main__':unittest.main()
