import jax
import tqdm

def get_tqdm_output(tqdm_refresh = 100):
    progress_bar = tqdm.tqdm(desc='Progress (-.--e---)')
    def update_progress(value, i, iterations):
        if int(i)==0:
            progress_bar.reset(int(iterations))
        progress_bar.n = int(i)
        progress_bar.desc = f'Progress ({float(value):.2e})'
        progress_bar.refresh()
    def true_fn(value, i, iterations):
        return jax.debug.callback(update_progress, value, i, iterations)
    false_fn = lambda *_ : None
    def output(value, i, iterations):
        return jax.lax.cond(jax.numpy.mod(i,tqdm_refresh) == 0, true_fn, false_fn, value, i, iterations)
    return output