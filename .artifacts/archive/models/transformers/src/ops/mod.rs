/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

mod merge;
mod split;

pub(crate) mod prelude {
    pub use super::merge::*;
    pub use super::split::*;
    pub(crate) use super::utils::*;
}

pub(crate) const ORDER: nd::Order = nd::Order::RowMajor;

pub(crate) mod utils {
    use concision::NdResult;
    use nd::prelude::*;
    use nd::{Data, Order, RemoveAxis};

    pub(crate) fn _merge<A, S, D>(
        arr: &ArrayBase<S, D>,
        src: usize,
        tgt: usize,
        order: Order,
    ) -> NdResult<Array<A, D::Smaller>>
    where
        A: Clone,
        D: RemoveAxis,
        S: Data<Elem = A>,
        D::Smaller: Dimension,
        ArrayBase<S, D>: Clone,
    {
        let shape = _merge_dim(&arr.raw_dim(), src);
        let mut head = arr.clone();
        head.swap_axes(src, tgt);
        head.to_shape((shape, order)).map(|x| x.to_owned())
    }

    pub(crate) fn _split<A, S, D, E>(
        arr: &ArrayBase<S, D>,
        h: usize,
        order: Order,
    ) -> NdResult<Array<A, E>>
    where
        A: Clone,
        D: Dimension<Larger = E>,
        E: RemoveAxis<Smaller = D>,
        S: Data<Elem = A>,
        ArrayBase<S, D>: Clone,
    {
        let src = if arr.ndim() >= 2 { arr.ndim() - 2 } else { 0 };
        let tgt = src + 1;
        let shape: E = _split_dim(&arr.raw_dim(), h);
        let mut head = arr.to_shape((shape, order))?.to_owned();
        head.swap_axes(src, tgt);
        Ok(head)
    }
    /// Creates the new dimension after merging two axes.
    pub(crate) fn _merge_dim<D>(dim: &D, axis: usize) -> D::Smaller
    where
        D: RemoveAxis,
        D::Smaller: Dimension,
    {
        // create a new dimension with one less axis; initialized with zeros
        let mut dn = <D as Dimension>::Smaller::zeros(dim.ndim() - 1);
        // create a mutable vector from the slice
        let mut shape = dim.slice().to_vec();
        // multiply the last axis by the target
        shape[dn.ndim()] *= shape[axis];
        // remove the last dimension
        shape.remove(axis);

        dn.slice_mut().copy_from_slice(&shape);
        dn
    }

    pub(crate) fn _split_dim<D>(dim: &D::Smaller, h: usize) -> D
    where
        D: RemoveAxis,
        D::Smaller: Dimension,
    {
        let rank = dim.ndim() + 1;
        // create a new dimension with one less axis; initialized with zeros
        let mut new_dim = D::zeros(rank);
        // create a mutable vector from the slice
        let mut shape = dim.slice().to_vec();
        // get and remove the last axis
        let bx = shape.pop().unwrap() / h;
        // extend the shape with the new axes
        shape.push(h);
        shape.push(bx);
        // shape.swap(rank - 2, rank - 3);
        // copy the values into the new dimension
        new_dim.slice_mut().copy_from_slice(&shape);
        new_dim
    }
}
