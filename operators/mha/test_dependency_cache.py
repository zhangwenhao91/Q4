import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from seed_dependency_cache import seed,DEPENDENCIES,sha256


class CacheTest(unittest.TestCase):
    def call(self,src,dst,spec):
        with contextlib.redirect_stdout(io.StringIO()): seed(src,dst,spec)

    def test_verified_copy_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'old';dst=Path(tmp)/'new'
            (src/'pkg').mkdir(parents=True)
            data=b'verified fixture';(src/'pkg'/'one.tar.gz').write_bytes(data)
            spec=[('one.tar.gz','one.tar.gz',hashlib.sha256(data).hexdigest())]
            self.call(src,dst,spec)
            self.assertEqual((dst/'one.tar.gz').read_bytes(),data)
            self.call(Path(tmp)/'missing-old',dst,spec)

    def test_wrong_hash_stops_before_any_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'old';dst=Path(tmp)/'new';src.mkdir()
            (src/'good').write_bytes(b'good');(src/'bad').write_bytes(b'corrupt')
            spec=[('good','good',hashlib.sha256(b'good').hexdigest()),
                  ('bad','bad',hashlib.sha256(b'expected').hexdigest())]
            with self.assertRaisesRegex(RuntimeError,'incomplete'): self.call(src,dst,spec)
            self.assertFalse(dst.exists())

    def test_archive_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'old';dst=Path(tmp)/'new';src.mkdir()
            (src/'protobuf-all-25.1.tar.gz').write_bytes(b'alias')
            self.call(src,dst,[('protobuf-25.1.tar.gz','protobuf-25.1.tar.gz',hashlib.sha256(b'alias').hexdigest())])
            self.assertEqual((dst/'protobuf-25.1.tar.gz').read_bytes(),b'alias')

    def test_bundled_real_archives_match_upstream(self):
        bundle=Path(__file__).with_name('dependency_archives')
        for name in ['eigen-5.0.0.tar.gz','abseil-cpp-20230802.1.tar.gz']:
            digest=next(h for n,_,h in DEPENDENCIES if n==name)
            self.assertEqual(sha256(bundle/name),digest)

    def test_use_bundle_when_old_cache_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle';bundle.mkdir()
            (bundle/'x.gz').write_bytes(b'fixture')
            spec=[('x.gz','x.gz',hashlib.sha256(b'fixture').hexdigest())]
            with contextlib.redirect_stdout(io.StringIO()):
                seed(Path(tmp)/'no-old-cache',Path(tmp)/'new',spec,bundle)
            self.assertEqual((Path(tmp)/'new'/'x.gz').read_bytes(),b'fixture')

    def test_target_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'escapes'):
                self.call(Path(tmp)/'old',Path(tmp)/'new',[('x','../outside','unused')])


if __name__=='__main__': unittest.main()
