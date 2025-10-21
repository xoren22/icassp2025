import time
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

from config import BASE_DIR
from inference import PathlossPredictor

def _generate_solution_df(model: PathlossPredictor, verbose=False, batch_size=8) -> pd.DataFrame:
    """
    Generate the solution DataFrame with predictions on all required
    (band, frequency, position) combinations, batching predictions in
    groups of `batch_size`.
    """
    # 1) Gather all sample dictionaries.
    samples = []
    for f_i in [1]:
        freq_mhz = 868
        for b in [1, 5]:
            for sp in range(25):
                sample_id_prefix = f"B{b}_Ant1_f{f_i}_S{sp}_"
                samples.append({
                    'freq_MHz': freq_mhz,
                    'sampling_position': sp,
                    'input_file': f"{BASE_DIR}/data/kaggle/Evaluation_Data_T1/Inputs/Task_1/B{b}_Ant1_f{f_i}_S{sp}.png",
                    'position_file': f"{BASE_DIR}/data/kaggle/Evaluation_Data_T1/Positions/Positions_B{b}_Ant1_f{f_i}.csv",
                    'radiation_pattern_file': f'{BASE_DIR}/data/kaggle/Evaluation_Data_T1/Radiation_Patterns/Ant1_Pattern.csv',
                    # We'll store the ID prefix for generating row labels later
                    'id_prefix': sample_id_prefix
                })

    # 2) Batch predict and build the solution rows.
    solution_parts = []
    total_samples = len(samples)
    if verbose:
        pbar = tqdm(range(0, total_samples, batch_size), total=(total_samples // batch_size), desc="Batch Prediction")
    else:
        pbar = range(0, total_samples, batch_size)

    for start_idx in pbar:
        batch = samples[start_idx:start_idx + batch_size]  # Next batch of size up to 8
        batch_preds = model.predict(batch)  # Returns a list of Tensors

        # 3) Convert each sample's output to a partial DataFrame
        for sample_dict, pred_tensor in zip(batch, batch_preds):
            pred_array = pred_tensor.cpu().detach().numpy()
            y_flat = np.expand_dims(pred_array.ravel(), axis=1)
            y_names = np.expand_dims(
                np.core.defchararray.add(
                    sample_dict['id_prefix'],
                    np.arange(pred_array.size).astype(str)
                ),
                1
            )
            # Combine ID and pathloss columns
            y_data = np.concatenate([y_names, y_flat], axis=1)
            part_df = pd.DataFrame(y_data, columns=["ID", "PL (dB)"])
            solution_parts.append(part_df)

    # 4) Concatenate all parts into a single DataFrame
    solution_df = pd.concat(solution_parts, ignore_index=True)
    return solution_df


def _submit_solution_to_kaggle(api: KaggleApi, file_path: str, competition: str, message: str):
    """
    Submits the CSV file to Kaggle and returns the submission object.
    """
    return api.competition_submit(file_path, message, competition)


def _poll_submission_score(api: KaggleApi, competition: str, submission) -> float:
    """
    Polls Kaggle until the submission completes and returns the public_score (MSE).
    """
    result = None
    while result is None:
        submission_results = api.competition_submissions(competition=competition)
        for sub in submission_results:
            # We look for the matching submission ref and a "COMPLETE" status
            if sub.ref == submission.ref and str(sub.status) == 'SubmissionStatus.COMPLETE':
                result = sub
                break
        time.sleep(5)  # Wait between checks

    return float(result.public_score)  # Kaggle's published MSE score


def _kaggle_eval_thread_fn(
    epoch: int,
    model,
    logger,
    csv_save_path: str = f"{BASE_DIR}/Task1.csv",
    competition: str = 'iprm-task-1',
    submission_message: str = "My auto submission",
):
    solution_df = _generate_solution_df(model)
    solution_df.to_csv(csv_save_path, index=False)

    api = KaggleApi()
    api.authenticate()
    submission = _submit_solution_to_kaggle(api, csv_save_path, competition, submission_message)
    kaggle_mse = _poll_submission_score(api, competition, submission)
    kaggle_rmse = kaggle_mse**0.5

    if logger is None:
        print(f"[Epoch {epoch}] Kaggle MSE: {kaggle_mse:.4f}, RMSE: {kaggle_rmse:.4f}")
    else:
        logger.log_kaggle_eval(kaggle_mse, epoch)


def kaggle_async_eval(
    epoch: int,
    model_ckpt_path=None,
    model=None,
    logger=None,
    csv_save_path: str = f"{BASE_DIR}/Task1.csv",
    competition: str = 'iprm-task-1',
    submission_message: str = "My auto submission",
):
    model = model or PathlossPredictor(model_ckpt_path=model_ckpt_path)
    thread = threading.Thread(
        target=_kaggle_eval_thread_fn,
        args=(epoch, model, logger, csv_save_path, competition, submission_message),
        daemon=False  # <--- Remove daemon
    )
    thread.start()



if __name__ == "__main__":
    model_ckpt_path = f'{BASE_DIR}/models/best_model.pth'
    from approx import Approx

    kaggle_async_eval(
        epoch=1,
        # model_ckpt_path=model_ckpt_path,
        model=Approx(),
    )

