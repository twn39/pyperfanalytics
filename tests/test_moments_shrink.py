import os
import json
import numpy as np
import pandas as pd
import pytest

import pyperfanalytics as pa
from pyperfanalytics.risk import (
    _m3_mat_to_vec,
    _m3_vec_to_mat,
    _m4_mat_to_vec,
    _m4_vec_to_mat,
    _m3_struct_Indep,
    _m3_struct_1f,
    _calc_M3_CC,
    _m3_struct_Simaan,
    _calc_M3_CCoefficients,
    _calc_M4_T12,
    _calc_M4_1f,
    _calc_M4_CC,
    _calc_M4_CCoefficients,
)


@pytest.fixture
def r_shrinkage_bench():
    path = "data/r_benchmarks_shrinkage.json"
    if not os.path.exists(path):
        pytest.skip("R benchmarks shrinkage not found. Run scripts/gen_r_bench_shrinkage.R first.")
    with open(path) as f:
        return json.load(f)

def test_m2_struct_cross_verification(edhec_data, r_shrinkage_bench):
    R = edhec_data.iloc[:, 0:3]
    f_single = edhec_data.iloc[:, 3]
    f_multi = edhec_data.iloc[:, 3:5]
    
    tol = 1e-6
    
    # 1. Indep
    expected = np.array(r_shrinkage_bench["m2_struct_Indep"])
    actual = pa.m2_struct(R, "Indep")
    np.testing.assert_allclose(actual, expected, atol=tol)
    
    # 2. IndepId
    expected = np.array(r_shrinkage_bench["m2_struct_IndepId"])
    actual = pa.m2_struct(R, "IndepId")
    np.testing.assert_allclose(actual, expected, atol=tol)
    
    # 3. observedfactor single
    expected = np.array(r_shrinkage_bench["m2_struct_observedfactor_single"])
    actual = pa.m2_struct(R, "observedfactor", f=f_single)
    np.testing.assert_allclose(actual, expected, atol=tol)
    
    # 4. observedfactor multi
    expected = np.array(r_shrinkage_bench["m2_struct_observedfactor_multi"])
    actual = pa.m2_struct(R, "observedfactor", f=f_multi)
    np.testing.assert_allclose(actual, expected, atol=tol)
    
    # 5. CC
    expected = np.array(r_shrinkage_bench["m2_struct_CC"])
    actual = pa.m2_struct(R, "CC")
    np.testing.assert_allclose(actual, expected, atol=tol)

def test_m3_struct_cross_verification(edhec_data, r_shrinkage_bench):
    R = edhec_data.iloc[:, 0:3]
    f_single = edhec_data.iloc[:, 3]
    f_multi = edhec_data.iloc[:, 3:5]
    
    tol = 1e-6
    
    # 1. latent1factor
    expected = np.array(r_shrinkage_bench["m3_struct_latent1factor"])
    actual = pa.m3_struct(R, "latent1factor", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 2. Indep
    expected = np.array(r_shrinkage_bench["m3_struct_Indep"])
    actual = pa.m3_struct(R, "Indep", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 3. IndepId
    expected = np.array(r_shrinkage_bench["m3_struct_IndepId"])
    actual = pa.m3_struct(R, "IndepId", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 4. observedfactor single
    expected = np.array(r_shrinkage_bench["m3_struct_observedfactor_single"])
    actual = pa.m3_struct(R, "observedfactor", f=f_single, as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 5. observedfactor multi
    expected = np.array(r_shrinkage_bench["m3_struct_observedfactor_multi"])
    actual = pa.m3_struct(R, "observedfactor", f=f_multi, as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 6. CC
    expected = np.array(r_shrinkage_bench["m3_struct_CC"])
    actual = pa.m3_struct(R, "CC", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 7. CS
    expected = np.array(r_shrinkage_bench["m3_struct_CS"])
    actual = pa.m3_struct(R, "CS", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 8. Unbiased Marginals
    expected = np.array(r_shrinkage_bench["m3_struct_Indep_unbiased"])
    actual = pa.m3_struct(R, "Indep", unbiasedMarg=True, as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    expected = np.array(r_shrinkage_bench["m3_struct_IndepId_unbiased"])
    actual = pa.m3_struct(R, "IndepId", unbiasedMarg=True, as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)

def test_m4_struct_cross_verification(edhec_data, r_shrinkage_bench):
    R = edhec_data.iloc[:, 0:3]
    f_single = edhec_data.iloc[:, 3]
    f_multi = edhec_data.iloc[:, 3:5]
    
    tol = 1e-6
    
    # 1. Indep
    expected = np.array(r_shrinkage_bench["m4_struct_Indep"])
    actual = pa.m4_struct(R, "Indep", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 2. IndepId
    expected = np.array(r_shrinkage_bench["m4_struct_IndepId"])
    actual = pa.m4_struct(R, "IndepId", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 3. observedfactor single
    expected = np.array(r_shrinkage_bench["m4_struct_observedfactor_single"])
    actual = pa.m4_struct(R, "observedfactor", f=f_single, as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 4. observedfactor multi
    expected = np.array(r_shrinkage_bench["m4_struct_observedfactor_multi"])
    actual = pa.m4_struct(R, "observedfactor", f=f_multi, as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)
    
    # 5. CC
    expected = np.array(r_shrinkage_bench["m4_struct_CC"])
    actual = pa.m4_struct(R, "CC", as_mat=False)
    np.testing.assert_allclose(actual.ravel(), expected.ravel(), atol=tol)

def test_m2_shrink_cross_verification(edhec_data, r_shrinkage_bench):
    R = edhec_data.iloc[:, 0:3]
    f_single = edhec_data.iloc[:, 3]
    
    tol = 1e-6
    
    # Single target (Analytical fast path)
    r_res = r_shrinkage_bench["m2_shrink_t1"]
    py_res = pa.m2_shrink(R, targets=1)
    np.testing.assert_allclose(py_res["M2sh"], np.array(r_res["M2sh"]), atol=tol)
    np.testing.assert_allclose(py_res["lambda"].ravel(), np.array([r_res["lambda"]]).ravel(), atol=tol)
    
    # Multi targets (SLSQP QP solver)
    r_res = r_shrinkage_bench["m2_shrink_tall"]
    py_res = pa.m2_shrink(R, targets=[1, 2, 3, 4], f=f_single)
    np.testing.assert_allclose(py_res["M2sh"], np.array(r_res["M2sh"]), atol=tol)
    np.testing.assert_allclose(py_res["lambda"].ravel(), np.array(r_res["lambda"]).ravel(), atol=tol)

def test_m3_shrink_cross_verification(edhec_data):
    R = edhec_data.iloc[:, 0:3]
    f_single = edhec_data.iloc[:, 3]
    
    tol = 1e-12
    T_obs = len(R)
    N_assets = R.shape[1]
    Xc = R.values - np.mean(R.values, axis=0, keepdims=True)
    Xc2 = Xc**2
    margvars = np.mean(Xc2, axis=0)
    margskews = np.mean(Xc**3, axis=0)
    margkurts = np.mean(Xc**4, axis=0)
    
    m11 = (Xc.T @ Xc) / T_obs
    m21 = (Xc2.T @ Xc) / T_obs
    m22 = (Xc2.T @ Xc2) / T_obs
    
    fc_kron = np.einsum('ti,tj->tij', Xc, Xc).reshape(T_obs, -1)
    M3_mat = (1.0 / T_obs) * (Xc.T @ fc_kron)
    M3 = _m3_mat_to_vec(M3_mat, N_assets)
    
    # 1. Single target (Analytical fast path)
    py_res = pa.m3_shrink(R, targets=1, as_mat=False)
    T1 = _m3_struct_Indep(margskews, N_assets)
    expected_lambda = np.clip(py_res["b"][0] / py_res["A"][0, 0], 0, 1)
    expected_M3sh = (1.0 - expected_lambda) * M3 + expected_lambda * T1
    
    np.testing.assert_allclose(py_res["lambda"][0], expected_lambda, atol=tol)
    np.testing.assert_allclose(py_res["M3sh"], expected_M3sh, atol=tol)
    
    # 2. Multi targets (SLSQP QP solver)
    py_res_multi = pa.m3_shrink(R, targets=[1, 2, 3, 4, 5, 6], f=f_single, as_mat=False)
    lambdas = py_res_multi["lambda"]
    assert np.sum(lambdas) <= 1.0001
    assert np.all(lambdas >= -1e-12)
    
    T1 = _m3_struct_Indep(margskews, N_assets)
    T2 = _m3_struct_Indep(np.repeat(np.mean(margskews), N_assets), N_assets)
    
    f_centered = f_single.values - np.mean(f_single.values)
    fvar_ddof1 = np.var(f_centered, ddof=1)
    fskew = np.mean(f_centered**3)
    beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
    T3 = _m3_struct_1f(margskews, beta, fskew, N_assets)
    
    r2, r4, r5 = _calc_M3_CCoefficients(Xc, margvars, margkurts, m21, m22, T_obs, N_assets)
    T4 = _calc_M3_CC(margvars, margskews, margkurts, r2, r4, r5, N_assets)
    
    margskewsroot = np.sign(margskews) * np.abs(margskews)**(1.0 / 3.0)
    T5 = _m3_struct_Simaan(margskewsroot, N_assets)
    
    T6 = np.zeros_like(M3)
    
    T_list = [T1, T2, T3, T4, T5, T6]
    expected_M3sh_multi = (1.0 - np.sum(lambdas)) * M3
    for tt, lam in enumerate(lambdas):
        expected_M3sh_multi += lam * T_list[tt]
        
    np.testing.assert_allclose(py_res_multi["M3sh"], expected_M3sh_multi, atol=tol)
    
    # 3. unbiasedMSE (uses VM3kstat)
    py_res_unbiased = pa.m3_shrink(R, targets=[1, 2, 6], unbiasedMSE=True, as_mat=False)
    lambdas_unb = py_res_unbiased["lambda"]
    assert np.sum(lambdas_unb) <= 1.0001
    assert np.all(lambdas_unb >= -1e-12)
    
    CC_unb = T_obs / ((T_obs - 1) * (T_obs - 2))
    M3_mat_unb = CC_unb * (Xc.T @ fc_kron)
    M3_unb = _m3_mat_to_vec(M3_mat_unb, N_assets)
    
    skews_unb = margskews * (T_obs**2) / ((T_obs - 1) * (T_obs - 2))
    T1_unb = _m3_struct_Indep(skews_unb, N_assets)
    T2_unb = _m3_struct_Indep(np.repeat(np.mean(skews_unb), N_assets), N_assets)
    T6_unb = np.zeros_like(M3_unb)
    
    expected_M3sh_unb = (1.0 - np.sum(lambdas_unb)) * M3_unb + lambdas_unb[0] * T1_unb + lambdas_unb[1] * T2_unb + lambdas_unb[2] * T6_unb
    np.testing.assert_allclose(py_res_unbiased["M3sh"], expected_M3sh_unb, atol=tol)

def test_m4_shrink_cross_verification(edhec_data):
    R = edhec_data.iloc[:, 0:3]
    f_single = edhec_data.iloc[:, 3]
    
    tol = 1e-12
    T_obs = len(R)
    N_assets = R.shape[1]
    Xc = R.values - np.mean(R.values, axis=0, keepdims=True)
    Xc2 = Xc**2
    margvars = np.mean(Xc2, axis=0)
    margkurts = np.mean(Xc**4, axis=0)
    marg6s = np.mean(Xc**6, axis=0)
    
    m11 = (Xc.T @ Xc) / T_obs
    m22 = (Xc2.T @ Xc2) / T_obs
    m31 = ((Xc**3).T @ Xc) / T_obs
    
    fc_kron = np.einsum('ti,tj,tk->tijk', Xc, Xc, Xc).reshape(T_obs, -1)
    M4_mat = (Xc.T @ fc_kron) / T_obs
    M4 = _m4_mat_to_vec(M4_mat, N_assets)
    
    # 1. Single target (Analytical fast path)
    py_res = pa.m4_shrink(R, targets=1, as_mat=False)
    T1 = _calc_M4_T12(margkurts, margvars, N_assets)
    expected_lambda = np.clip(py_res["b"][0] / py_res["A"][0, 0], 0, 1)
    expected_M4sh = (1.0 - expected_lambda) * M4 + expected_lambda * T1
    
    np.testing.assert_allclose(py_res["lambda"][0], expected_lambda, atol=tol)
    np.testing.assert_allclose(py_res["M4sh"], expected_M4sh, atol=tol)
    
    # 2. Multi targets (SLSQP QP solver)
    py_res_multi = pa.m4_shrink(R, targets=[1, 2, 3, 4], f=f_single, as_mat=False)
    lambdas = py_res_multi["lambda"]
    assert np.sum(lambdas) <= 1.0001
    assert np.all(lambdas >= -1e-12)
    
    T1 = _calc_M4_T12(margkurts, margvars, N_assets)
    
    meanmargkurts = np.mean(margkurts)
    meank_iikk = np.sqrt(np.mean(margvars**2))
    T2 = _calc_M4_T12(np.repeat(meanmargkurts, N_assets), np.repeat(meank_iikk, N_assets), N_assets)
    
    f_centered = f_single.values - np.mean(f_single.values)
    fvar = np.mean(f_centered**2)
    fvar_ddof1 = np.var(f_centered, ddof=1)
    fkurt = np.mean(f_centered**4)
    beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
    epsvars = margvars - beta**2 * fvar
    T3 = _calc_M4_1f(margkurts, fvar, fkurt, epsvars, beta, N_assets)
    
    r3, r5, r6, r7 = _calc_M4_CCoefficients(Xc, margvars, margkurts, marg6s, m22, m31, T_obs, N_assets)
    T4 = _calc_M4_CC(margvars, margkurts, marg6s, r3, r5, r6, r7, N_assets)
    
    T_list = [T1, T2, T3, T4]
    expected_M4sh_multi = (1.0 - np.sum(lambdas)) * M4
    for tt, lam in enumerate(lambdas):
        expected_M4sh_multi += lam * T_list[tt]
        
    np.testing.assert_allclose(py_res_multi["M4sh"], expected_M4sh_multi, atol=tol)

def test_moments_edge_cases(edhec_data):
    # Single asset should fail
    R_single = edhec_data.iloc[:, [0]]
    with pytest.raises(ValueError, match="at least 2 variables"):
        pa.m2_struct(R_single)
        
    with pytest.raises(ValueError, match="at least 2 variables"):
        pa.m2_shrink(R_single)
        
    # Providing no targets
    R = edhec_data.iloc[:, 0:3]
    with pytest.raises(ValueError, match="No targets selected"):
        pa.m2_shrink(R, targets=[])
        
    # Invalid target type
    with pytest.raises(ValueError, match="Select valid targets"):
        pa.m2_shrink(R, targets=[99])
        
    # Missing factor for observedfactor
    with pytest.raises(ValueError, match="Provide factor observations"):
        pa.m2_struct(R, struct="observedfactor")
        
    with pytest.raises(ValueError, match="Provide factor observations"):
        pa.m2_shrink(R, targets=3)
        
    # UnbiasedMSE combined with T3/T4/T5
    with pytest.raises(ValueError, match="unbiasedMSE can only be combined"):
        pa.m3_shrink(R, targets=[1, 3], f=edhec_data.iloc[:, 3], unbiasedMSE=True)

def test_tensor_conversion():
    # Test M3 conversion roundtrip
    np.random.seed(42)
    N = 4
    ncosk = N * (N + 1) * (N + 2) // 6
    vec = np.random.normal(0, 1, ncosk)
    mat = _m3_vec_to_mat(vec, N)
    vec_back = _m3_mat_to_vec(mat, N)
    np.testing.assert_allclose(vec_back, vec)
    
    # Test M4 conversion roundtrip
    ncokurt = N * (N + 1) * (N + 2) * (N + 3) // 24
    vec4 = np.random.normal(0, 1, ncokurt)
    mat4 = _m4_vec_to_mat(vec4, N)
    vec4_back = _m4_mat_to_vec(mat4, N)
    np.testing.assert_allclose(vec4_back, vec4)
