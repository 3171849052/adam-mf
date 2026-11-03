"""Fresh solving, mathematical checks and cache integrity; no historical inputs."""
import numpy as np
import pytest
from adam_mf.audit import read_json, atomic_json, file_hash
from adam_mf.privacy import matrix_design as design

@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    folder=tmp_path_factory.mktemp('computed-matrices')
    design.prepare_matrices(folder,horizons=(10,))
    return folder


def test_fresh_design_noise_covariance_and_objective(generated):
    for work,mechanism in design.PAIRS:
        arrays=design.load_matrix(generated,work,mechanism,10,.9)
        A,C,D=arrays['A'],arrays['C'],arrays['D'];meta=arrays['metadata']
        np.testing.assert_allclose(C@D,np.eye(10),atol=1e-12)
        np.testing.assert_allclose((A@D)@C,A,atol=1e-12)
        sigma=meta['audit']['sigma_sum_clip1']
        np.testing.assert_allclose(C@arrays['covariance_clip1']@C.T,sigma**2*np.eye(10),atol=1e-10)
        if mechanism=='iid':np.testing.assert_array_equal(D,np.eye(10))
        else:
            assert meta['selected_error']<=meta['initial_error']
            assert np.any(D!=np.eye(10))
        assert design.verify_generated(generated,work,mechanism,10)['epsilon']==pytest.approx(8.)


def test_reuse_does_not_solve(generated,monkeypatch):
    before={p.name:file_hash(p) for p in generated.glob('*.npz')}
    def forbidden(*args):raise AssertionError('Validated matrices must be reused')
    monkeypatch.setattr(design,'optimize',forbidden)
    design.prepare_matrices(generated,horizons=(10,))
    assert before=={p.name:file_hash(p) for p in generated.glob('*.npz')}


def test_corruption_and_changed_recipe_rejected(generated,tmp_path):
    import shutil
    folder=tmp_path/'matrices';shutil.copytree(generated,folder)
    p=folder/'sgd_b2lt_T10.npz';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):design.verify_generated(folder,'sgd','b2lt',10)
    shutil.copy2(generated/p.name,p)
    m=p.with_suffix('.json');meta=read_json(m);meta['recipe']['Q']=99;atomic_json(m,meta)
    with pytest.raises(ValueError,match='recipe changed'):design.prepare_matrices(folder,horizons=(10,))


def test_small_design_repeatability(generated):
    saved=design.load_matrix(generated,'sgd','b2lt',10,.9)
    C,D,d,meta=design.optimize(saved['A'],'b2lt',2,'sgd')
    np.testing.assert_array_equal(C,saved['C'])
    np.testing.assert_array_equal(D,saved['D'])
    assert meta['selected_parameters']==saved['metadata']['selected_parameters']


def test_fetch_assets_uses_solver_without_archive(tmp_path,monkeypatch):
    import adam_mf.assets as assets
    from config_helpers import load_config
    c=load_config();events=[]
    monkeypatch.setattr(assets,'catalog',lambda:dict(files=[]))
    monkeypatch.setattr(assets,'local_path',lambda p:tmp_path/p)
    monkeypatch.setattr(assets,'manifests',lambda *a,**k:events.append('manifest'))
    monkeypatch.setattr(assets,'verify_assets',lambda *a,**k:{})
    monkeypatch.setattr(design,'prepare_matrices',lambda folder:events.append(folder))
    assets.fetch_assets(c)
    assert events==[tmp_path/c['paths']['matrices'],'manifest']
