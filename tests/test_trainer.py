import pytest

from ncalab import EarlyStopping


def test_initial_state():
    es = EarlyStopping(patience=3)
    assert es.patience == 3
    assert es.min_delta == pytest.approx(1e-6)
    assert es.best_accuracy == 0.0
    assert es.counter == 0
    assert not es.done()

def test_reset():
    es = EarlyStopping(patience=2)
    es.counter = 1
    es.step(0.8)
    assert es.best_accuracy == pytest.approx(0.8)
    assert es.counter == 0
    assert not es.done()

def test_done():
    es = EarlyStopping(patience=2)
    es.step(0.8)
    assert not es.done()
    es.step(0.8)
    assert es.counter == 1
    assert not es.done()
    es.step(0.8)
    assert es.counter == 2
    assert es.done()

def test_min_delta():
    es = EarlyStopping(patience=3, min_delta=0.1)
    es.step(0.5)
    assert es.best_accuracy == pytest.approx(0.5)
    assert es.counter == 0
    es.step(0.55)
    assert es.best_accuracy == pytest.approx(0.5)
    assert es.counter == 1
    es.step(0.6)
    assert es.best_accuracy == pytest.approx(0.6)
    assert es.counter == 0
