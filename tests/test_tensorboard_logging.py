"""Verify optional event output can be read by TensorBoard."""
import pytest

from radargen.training.tensorboard_logging import log_scalars, tensorboard_writer


def test_disabled_does_not_create_directory(tmp_path):
    with tensorboard_writer(tmp_path / 'unused') as writer:
        assert writer is None
        log_scalars(writer, 'train', {'loss': 1.0}, 1)
    assert not (tmp_path / 'unused').exists()


def test_events_and_exception_cleanup(tmp_path):
    pytest.importorskip('torch.utils.tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with pytest.raises(RuntimeError, match='stop'):
        with tensorboard_writer(tmp_path, True) as writer:
            log_scalars(writer, 'validation', {
                'loss': 2.5, 'nested': {'mean': 3.0}, 'missing': None,
                'invalid': float('nan'), 'label': 'sample', 'complete': True}, 7)
            raise RuntimeError('stop')
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    assert set(events.Tags()['scalars']) == {'validation/loss', 'validation/nested/mean'}
    scalar = events.Scalars('validation/loss')[0]
    assert scalar.step == 7
    assert scalar.value == 2.5


def test_figure_event(tmp_path):
    pytest.importorskip('torch.utils.tensorboard')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    with tensorboard_writer(tmp_path, True) as writer:
        writer.add_figure('prediction/sample_0', fig, 3)
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    assert events.Images('prediction/sample_0')[0].step == 3
    assert not plt.fignum_exists(fig.number)


def test_training_emits_events(tmp_path):
    pytest.importorskip('torch.utils.tensorboard')
    from types import SimpleNamespace
    from test_dino_ppp_training import small_training_config, TinyDirectPPP, TinyBEVDataset
    from scripts.train_ppp import run_training
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    config = small_training_config()
    config.work_dir = str(tmp_path)
    config.tensorboard = True
    options = SimpleNamespace(smoke_check=False, max_updates=1, resume=None)
    components = SimpleNamespace(model=TinyDirectPPP(), condition_encoder=None)
    result = run_training(config, options, components=components, dataset=TinyBEVDataset())
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    assert events.Scalars('train/loss_total')[0].step == result['progress']['global_step']
