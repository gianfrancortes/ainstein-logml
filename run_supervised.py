import tensorflow as tf
import yaml
import os

tfk = tf.keras
tfk.backend.set_floatx("float64")

from geometry.ball import AnalyticMetric_Ball, PatchChange_Coordinates_Ball
from helper_functions.helper_functions import cholesky_to_vec
from network.ball import BallGlobalModel
from sampling.ball import BallSample, CubeSample


def main(identity_bool=False, hp=None):
    ###########################################################################
    ### Data set-up ###
    if hp["ball"]:
        train_sample = BallSample(
            hp["num_samples"], hp["dim"], hp["patch_width"], hp["density_power"]
        )
        if hp["validate"]:
            val_sample = BallSample(
                hp["num_val_samples"], hp["dim"], hp["patch_width"], hp["density_power"]
            )
    else:
        assert hp["n_patches"] == 1
        train_sample = CubeSample(
            hp["num_samples"], hp["dim"], hp["patch_width"], hp["density_power"]
        )
        if hp["validate"]:
            val_sample = CubeSample(
                hp["num_val_samples"], hp["dim"], hp["patch_width"], hp["density_power"]
            )

    train_sample_inputs = [train_sample]
    if hp["n_patches"] == 2:
        train_sample_inputs.append(PatchChange_Coordinates_Ball(train_sample))
    elif hp["n_patches"] > 2:
        raise SystemExit("codebase not yet configured for >2 patches...")
    train_sample_metrics = [
        AnalyticMetric_Ball(ts, identity=identity_bool) for ts in train_sample_inputs
    ]

    if hp["validate"]:
        val_sample_inputs = [val_sample]
        if hp["n_patches"] > 1:
            val_sample_inputs.append(PatchChange_Coordinates_Ball(val_sample))
        val_sample_metrics = [
            AnalyticMetric_Ball(vs, identity=identity_bool) for vs in val_sample_inputs
        ]

    train_sample_metrics_vecs = [cholesky_to_vec(tsm) for tsm in train_sample_metrics]
    if hp["validate"]:
        val_sample_metrics_vecs = [cholesky_to_vec(vsm) for vsm in val_sample_metrics]

    train_sample_tf = tf.convert_to_tensor(train_sample, dtype=tf.float64)
    train_sample_metrics_tf = tf.convert_to_tensor(
        tf.concat(train_sample_metrics_vecs, axis=1), dtype=tf.float64
    )
    if hp["validate"]:
        val_sample_tf = tf.convert_to_tensor(val_sample, dtype=tf.float64)
        val_sample_metrics_tf = tf.convert_to_tensor(
            tf.concat(val_sample_metrics_vecs, axis=1), dtype=tf.float64
        )
        val_data = (val_sample_tf, val_sample_metrics_tf)
    else:
        val_data = None

    ###########################################################################
    ### Model set-up ###
    if hp["init_learning_rate"] == hp["min_learning_rate"]:
        optimiser = tfk.optimizers.Adam(learning_rate=hp["init_learning_rate"])
    else:
        lr_schedule = tfk.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=hp["init_learning_rate"],
            decay_steps=1000,
            end_learning_rate=hp["min_learning_rate"],
            power=1.0,
        )
        optimiser = tfk.optimizers.Adam(learning_rate=lr_schedule)

    if hp["saved_model"]:
        model = tfk.models.load_model(hp["saved_model_path"])
        model.compile(optimizer=optimiser, loss="MSE")
        # update hps from imported model
        hp.update({k: model.hp[k] for k in ["dim", "n_patches", "n_hidden", "n_layers", "activations", "use_bias"]})
        model.hp = hp
        model.set_serializable_hp()
    else:
        model = BallGlobalModel(hp)
        model.compile(optimizer=optimiser, loss="MSE")

    ###########################################################################
    ### Training ###
    loss_hist = model.fit(
        train_sample_tf,
        train_sample_metrics_tf,
        batch_size=hp["batch_size"],
        epochs=hp["epochs"],
        verbose=hp["verbosity"],
        validation_data=val_data,
        shuffle=True,
    )

    return model, loss_hist, train_sample_tf, train_sample_metrics_tf, val_data


if __name__ == "__main__":
    with open("hyperparameters/hps.yaml", "r") as file:
        hp = yaml.safe_load(file)

    identity_bool = True
    save = True
    save_flag = hp["save_flag"]
    save_path = hp.get("saved_model_path", f"runs_supervised/supervised_model_{save_flag}.keras")

    # Train
    network, lh, train_coords, train_metrics, val_data = main(identity_bool, hp)
    print("trained.....")

    if save:
        print(f"[DEBUG] Will save to: {save_path}")
        network.save(save_path)
