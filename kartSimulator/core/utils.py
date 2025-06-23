import math




def update_scheduler(t, current_val, last_update, *,
                     start_t, end_t, min_val, max_val,
                     step=100):
    """
    t             : current timestep (integer)
    current_val   : scheduler's current value (float)
    last_update   : timestep when it was last updated (int or None)
    start_t       : timestep to begin ramping (inclusive)
    end_t         : timestep when it should have reached max_val (inclusive)
    min_val       : value before start_t
    max_val       : capped value at or after end_t
    step          : how many timesteps per “chunk” of increment

    Returns (new_val, new_last_update). If t < start_t, nothing changes.
    Once t >= start_t, it will add (
      (max_val - min_val) * step / (end_t - start_t)
    ) every time we see a full ‘step’ worth of timesteps since last_update.
    After t >= end_t, it just clamps to max_val.
    """
    # before the ramp starts, stay at min_val
    if t < start_t:
        return current_val, last_update

    # once we're past the ramp interval, lock to max_val
    if t >= end_t:
        return max_val, last_update

    # compute how much to add every “step” timesteps
    total_range = end_t - start_t
    if total_range <= 0:
        # avoid division by zero; just clamp
        return max_val, last_update

    per_chunk = (max_val - min_val) * step / total_range

    # initialize last_update to the moment we began ramping
    if last_update is None:
        # if we want exactly one increment only after step
        # we could set last_update = t. But if you prefer “catching up”
        # with all missed chunks since start_t, do last_update = start_t.
        # Here we mirror the earlier behavior (no catch‐up on first jump).
        last_update = t

    # figure out how many full steps have passed since last_update
    elapsed = t - last_update
    if elapsed >= step:
        chunks = elapsed // step
        new_val = current_val + per_chunk * chunks
        if new_val > max_val:
            new_val = max_val
        new_last = last_update + int(chunks * step)
        return new_val, new_last

    return current_val, last_update