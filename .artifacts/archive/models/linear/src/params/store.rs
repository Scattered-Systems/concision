/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Biased, Features, Node, ParamMode, Unbiased, build_bias};
use concision::dimensional;
use core::marker::PhantomData;
use nd::*;
use num::{One, Zero};

/// [ParamsBase] is a flexible store for linear parameters; it is parameterized to accept
/// a `K` type, used to designate the store as either [Biased](crate::Biased) or [Unbiased](crate::Unbiased).
/// This was done in an effort to streamline the creation of new instances of the store, and to provide
/// a more ergonomic interface for the user. The store is also equipped with a number of builder methods
///  native to the [ArrayBase] from `ndarray`.
pub struct ParamsBase<S = OwnedRepr<f64>, D = Ix2, K = Biased>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) bias: Option<ArrayBase<S, D::Smaller>>,
    pub(crate) weight: ArrayBase<S, D>,
    pub(crate) _mode: PhantomData<K>,
}

impl<A, S, D, K> ParamsBase<S, D, K>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    pub fn from_elem<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        K: ParamMode,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        let bias = if K::BIASED {
            Some(ArrayBase::from_elem(
                crate::bias_dim(dim.clone()),
                elem.clone(),
            ))
        } else {
            None
        };
        Self {
            bias,
            weight: ArrayBase::from_elem(dim, elem),
            _mode: PhantomData::<K>,
        }
    }

    pub fn into_biased(self) -> ParamsBase<S, D, Biased>
    where
        A: Default,
        K: 'static,
        S: DataOwned,
    {
        if self.is_biased() {
            return ParamsBase {
                bias: self.bias,
                weight: self.weight,
                _mode: PhantomData::<Biased>,
            };
        }
        let sm = crate::bias_dim(self.raw_dim());
        ParamsBase {
            bias: Some(ArrayBase::default(sm)),
            weight: self.weight,
            _mode: PhantomData::<Biased>,
        }
    }

    pub fn into_unbiased(self) -> ParamsBase<S, D, Unbiased> {
        ParamsBase {
            bias: None,
            weight: self.weight,
            _mode: PhantomData::<Unbiased>,
        }
    }

    pub const fn weights(&self) -> &ArrayBase<S, D> {
        &self.weight
    }

    pub fn weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.weight
    }

    pub fn features(&self) -> Features {
        Features::from_shape(self.shape())
    }

    pub fn in_features(&self) -> usize {
        self.features().dmodel()
    }

    pub fn out_features(&self) -> usize {
        if self.ndim() == 1 {
            return 1;
        }
        self.shape()[1]
    }
    /// Returns true if the parameter store is biased;
    /// Compares the [TypeId](core::any::TypeId) of the store with the [Biased](crate::Biased) type.
    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        crate::is_biased::<K>()
    }

    pbuilder!(new.default where A: Default, S: DataOwned);

    pbuilder!(ones where A: Clone + One, S: DataOwned);

    pbuilder!(zeros where A: Clone + Zero, S: DataOwned);

    dimensional!(weight);

    wnbview!(into_owned::<OwnedRepr>(self) where A: Clone, S: Data);

    wnbview!(into_shared::<OwnedArcRepr>(self) where A: Clone, S: DataOwned);

    wnbview!(to_owned::<OwnedRepr>(&self) where A: Clone, S: Data);

    wnbview!(to_shared::<OwnedArcRepr>(&self) where A: Clone, S: DataOwned);

    wnbview!(view::<'a, ViewRepr>(&self) where A: Clone, S: Data);

    wnbview!(view_mut::<'a, ViewRepr>(&mut self) where A: Clone, S: DataMut);
}

impl<A, S, D> ParamsBase<S, D, Biased>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    /// Create a new biased parameter store from the given shape.
    pub fn biased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        Self {
            bias: build_bias(true, dim.clone(), ArrayBase::default),
            weight: ArrayBase::default(dim),
            _mode: PhantomData::<Biased>,
        }
    }
    /// Return an unwraped, immutable reference to the bias array.
    pub fn bias(&self) -> &ArrayBase<S, D::Smaller> {
        self.bias.as_ref().unwrap()
    }
    /// Return an unwraped, mutable reference to the bias array.
    pub fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        self.bias.as_mut().unwrap()
    }
}

impl<A, S, D> ParamsBase<S, D, Unbiased>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Create a new unbiased parameter store from the given shape.
    pub fn unbiased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            bias: None,
            weight: ArrayBase::default(shape),
            _mode: PhantomData::<Unbiased>,
        }
    }
}
impl<A, S, K> ParamsBase<S, Ix2, K>
where
    K: 'static,
    S: RawData<Elem = A>,
{
    pub fn set_node(&mut self, idx: usize, node: Node<A>)
    where
        A: Clone + Default,
        S: DataMut + DataOwned,
    {
        let (weight, bias) = node;
        if let Some(bias) = bias {
            if !self.is_biased() {
                let mut tmp = ArrayBase::default(self.out_features());
                tmp.index_axis_mut(Axis(0), idx).assign(&bias);
                self.bias = Some(tmp);
            }
            self.bias
                .as_mut()
                .unwrap()
                .index_axis_mut(Axis(0), idx)
                .assign(&bias);
        }

        self.weights_mut()
            .index_axis_mut(Axis(0), idx)
            .assign(&weight);
    }
}

impl<A, S, D> Default for ParamsBase<S, D, Biased>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self {
            bias: Some(Default::default()),
            weight: Default::default(),
            _mode: PhantomData::<Biased>,
        }
    }
}

impl<A, S, D> Default for ParamsBase<S, D, Unbiased>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self {
            bias: None,
            weight: Default::default(),
            _mode: PhantomData::<Unbiased>,
        }
    }
}
