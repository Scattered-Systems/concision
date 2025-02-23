/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{PadAction, PadError, PadMode};
use crate::ArrayLike;
use ndarray::{Array, ArrayBase, AxisDescription, Data, DataOwned, Dimension, Slice};
use num::{FromPrimitive, Num};

fn reader(nb_dim: usize, pad: &[[usize; 2]]) -> Result<Vec<[usize; 2]>, PadError> {
    if pad.len() == 1 && pad.len() < nb_dim {
        // The user provided a single padding for all dimensions
        Ok(vec![pad[0]; nb_dim])
    } else if pad.len() == nb_dim {
        Ok(pad.to_vec())
    } else {
        Err(PadError::InconsistentDimensions(String::new()))
    }
}

pub fn pad<A, S, D>(
    data: &ArrayBase<S, D>,
    pad: &[[usize; 2]],
    mode: PadMode<A>,
) -> Result<Array<A, D>, PadError>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    let pad = reader(data.ndim(), pad)?;
    let mut dim = data.raw_dim();
    for (ax, (&ax_len, pad)) in data.shape().iter().zip(pad.iter()).enumerate() {
        dim[ax] = ax_len + pad[0] + pad[1];
    }

    let mut padded = data.array_like(dim, mode.init()).to_owned();
    let _ = pad_to(data, &pad, mode, &mut padded)?;
    Ok(padded)
}

pub fn pad_to<A, S, D>(
    data: &ArrayBase<S, D>,
    pad: &[[usize; 2]],
    mode: PadMode<A>,
    output: &mut Array<A, D>,
) -> super::PadResult
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: Data<Elem = A>,
{
    let pad = reader(data.ndim(), pad)?;

    // Select portion of padded array that needs to be copied from the original array.
    output
        .slice_each_axis_mut(|ad| {
            let AxisDescription { axis, len, .. } = ad;
            let pad = pad[axis.index()];
            Slice::from(pad[0]..len - pad[1])
        })
        .assign(data);

    match mode.into_pad_action() {
        PadAction::StopAfterCopy => {
            // Do nothing
            return Ok(());
        }
        _ => unimplemented!(),
    }
}
