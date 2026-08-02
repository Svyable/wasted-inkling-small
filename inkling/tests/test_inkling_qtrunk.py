#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ctypes, json, shutil, subprocess, sys, tempfile, unittest, zlib
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools')); sys.path.insert(0,str(REPO/'tests'))
from inkling_qtrunk import HEADER, FMT_Q4G, FMT_Q8G, QTrunkError, quantize_trunk, verify_qtensor
from inkling_trunk import stage_trunk
from test_inkling_trunk import make_full_checkpoint

class QT(ctypes.Structure):
    _fields_=[('fd',ctypes.c_int),('fmt',ctypes.c_int),('ndim',ctypes.c_int),('group',ctypes.c_int),
              ('shape',ctypes.c_int*4),('rows',ctypes.c_uint64),('cols',ctypes.c_uint64),
              ('qbytes',ctypes.c_uint64),('scale_bytes',ctypes.c_uint64),('scale_off',ctypes.c_uint64),
              ('rowbytes',ctypes.c_size_t),('groups',ctypes.c_size_t),('q_crc32',ctypes.c_uint32),
              ('scale_crc32',ctypes.c_uint32),('verify_crc',ctypes.c_int),
              ('qrow',ctypes.POINTER(ctypes.c_ubyte)),('scales',ctypes.POINTER(ctypes.c_uint16)),
              ('qdata',ctypes.POINTER(ctypes.c_ubyte)),('scale_data',ctypes.POINTER(ctypes.c_uint16))]

class QTrunkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc=shutil.which('cc')
        if not cc: raise unittest.SkipTest('C compiler unavailable')
        cls.bt=tempfile.TemporaryDirectory(); so=Path(cls.bt.name)/'libqt.so'
        subprocess.run([cc,'-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',f'-I{REPO/"src"}',str(REPO/'src/inkling_qtensor.c'),'-o',str(so)],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(so))
        cls.lib.waste_inkling_qtensor_open.argtypes=[ctypes.POINTER(QT),ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
        cls.lib.waste_inkling_qtensor_open.restype=ctypes.c_int
        cls.lib.waste_inkling_qtensor_close.argtypes=[ctypes.POINTER(QT)]
        cls.lib.waste_inkling_qtensor_row.argtypes=[ctypes.POINTER(QT),ctypes.c_uint64,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]
        cls.lib.waste_inkling_qtensor_matvec.argtypes=[ctypes.POINTER(QT),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_size_t]
        cls.lib.waste_inkling_qtensor_matvec_rows.argtypes=[ctypes.POINTER(QT),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t]
        cls.lib.waste_inkling_qtensor_load_resident.argtypes=[ctypes.POINTER(QT)]
        cls.lib.waste_inkling_qtensor_load_resident.restype=ctypes.c_int
        cls.lib.waste_inkling_qtensor_resident_bytes.argtypes=[ctypes.POINTER(QT)]
        cls.lib.waste_inkling_qtensor_resident_bytes.restype=ctypes.c_size_t
    @classmethod
    def tearDownClass(cls): cls.bt.cleanup()

    def setup_stage(self):
        td,src,_=make_full_checkpoint(); self.addCleanup(td.cleanup)
        ot=tempfile.TemporaryDirectory(); self.addCleanup(ot.cleanup); out=Path(ot.name)
        stage_trunk(src,out,chunk_bytes=128); return out

    def _open(self,out,meta):
        t=QT(); shape=(ctypes.c_int*len(meta['shape']))(*meta['shape'])
        path=out/'qtrunk-stage'/meta['file']
        self.assertEqual(self.lib.waste_inkling_qtensor_open(ctypes.byref(t),str(path).encode(),meta['target'].encode(),shape,len(meta['shape']),1),0)
        return t

    def test_q8_and_q4_c_rows_and_matvec_match_python_dequantization(self):
        for bits in (8,4):
            with self.subTest(bits=bits):
                out=self.setup_stage(); stage=quantize_trunk(out,bits=bits,group=8,chunk_rows=3)
                meta=next(x for x in stage['tensors'] if x['target']=='inkling.layer.0.q')
                t=self._open(out,meta)
                try:
                    rows,cols=meta['rows'],meta['cols']; path=out/'qtrunk-stage'/meta['file']; raw=path.read_bytes()
                    q=raw[HEADER.size:HEADER.size+meta['qbytes']]
                    scales=torch.frombuffer(bytearray(raw[HEADER.size+meta['qbytes']:HEADER.size+meta['qbytes']+meta['scale_bytes']]),dtype=torch.float16).float().reshape(rows,-1)
                    if bits==8:
                        qi=torch.frombuffer(bytearray(q),dtype=torch.int8).float().reshape(rows,-1)
                    else:
                        b=torch.frombuffer(bytearray(q),dtype=torch.uint8); qi=torch.stack(((b&15).to(torch.int16)-8,(b>>4).to(torch.int16)-8),dim=1).reshape(rows,-1).float()
                    ref=(qi.reshape(rows,-1,8)*scales[:,:,None]).reshape(rows,-1)[:,:cols]
                    row=(ctypes.c_float*cols)(); self.assertEqual(self.lib.waste_inkling_qtensor_row(ctypes.byref(t),1,row,cols),0)
                    torch.testing.assert_close(torch.tensor(list(row)),ref[1],rtol=1e-6,atol=1e-6)
                    x=torch.linspace(-1,1,cols); xin=(ctypes.c_float*cols)(*x.tolist()); y=(ctypes.c_float*rows)()
                    self.assertEqual(self.lib.waste_inkling_qtensor_matvec(ctypes.byref(t),xin,y,rows,cols),0)
                    torch.testing.assert_close(torch.tensor(list(y)),ref@x,rtol=2e-5,atol=2e-5)
                finally: self.lib.waste_inkling_qtensor_close(ctypes.byref(t))

    def test_resident_payload_and_row_range_match_disk_path(self):
        out=self.setup_stage(); stage=quantize_trunk(out,bits=4,group=8,chunk_rows=2)
        meta=next(x for x in stage['tensors'] if x['target']=='inkling.layer.0.q')
        t=self._open(out,meta)
        try:
            rows,cols=meta['rows'],meta['cols']
            x=torch.linspace(-0.5,0.5,cols); xin=(ctypes.c_float*cols)(*x.tolist())
            disk=(ctypes.c_float*2)()
            self.assertEqual(self.lib.waste_inkling_qtensor_matvec_rows(
                ctypes.byref(t),xin,disk,1,2,cols),0)
            self.assertEqual(self.lib.waste_inkling_qtensor_load_resident(ctypes.byref(t)),0)
            self.assertEqual(self.lib.waste_inkling_qtensor_resident_bytes(ctypes.byref(t)),
                             meta['qbytes']+meta['scale_bytes'])
            resident=(ctypes.c_float*2)()
            self.assertEqual(self.lib.waste_inkling_qtensor_matvec_rows(
                ctypes.byref(t),xin,resident,1,2,cols),0)
            self.assertEqual(bytes(disk),bytes(resident))
            self.assertNotEqual(bool(t.qdata),False)
        finally: self.lib.waste_inkling_qtensor_close(ctypes.byref(t))

    def test_sensitive_policy_and_resume(self):
        out=self.setup_stage(); first=quantize_trunk(out,bits=4,group=8,chunk_rows=2)
        by={x['target']:x for x in first['tensors']}
        self.assertEqual(by['inkling.embed']['format'],FMT_Q8G)
        self.assertEqual(by['inkling.layer.1.router.weight']['format'],FMT_Q8G)
        self.assertEqual(by['inkling.layer.0.q']['format'],FMT_Q4G)
        second=quantize_trunk(out,bits=4,group=8,chunk_rows=2)
        self.assertTrue(all(x['reused'] for x in second['tensors']))
        self.assertEqual(second['totals']['source_bytes_read'],0)

    def test_crc_corruption_is_rejected(self):
        out=self.setup_stage(); stage=quantize_trunk(out,bits=8,group=8)
        meta=stage['tensors'][0]; path=out/'qtrunk-stage'/meta['file']
        with path.open('r+b') as f:
            f.seek(HEADER.size+2); b=f.read(1); f.seek(HEADER.size+2); f.write(bytes([b[0]^1]))
        with self.assertRaisesRegex(QTrunkError,'CRC mismatch'):
            verify_qtensor(path,meta)

if __name__=='__main__': unittest.main()
