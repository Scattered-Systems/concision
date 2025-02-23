/*
   Appellation: plan <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::slice;

use super::utils::fft_permutation;

#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Deserialize, serde_derive::Serialize)
)]
pub struct FftPlan {
    len: usize,
    plan: Vec<usize>,
}

impl FftPlan {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            plan: Vec::with_capacity(len),
        }
    }

    pub fn build(self) -> Self {
        let plan = fft_permutation(self.len);
        Self { plan, ..self }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.plan.clear();
    }

    pub fn get(&self, index: usize) -> Option<&usize> {
        self.plan().get(index)
    }

    pub fn iter(&self) -> slice::Iter<usize> {
        self.plan().iter()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn plan(&self) -> &[usize] {
        &self.plan
    }

    pub fn set(&mut self, len: usize) {
        self.len = len;
        self.plan = Vec::with_capacity(len);
    }

    pub fn with(self, len: usize) -> Self {
        Self {
            len,
            plan: Vec::with_capacity(len),
        }
    }
}

impl AsRef<[usize]> for FftPlan {
    fn as_ref(&self) -> &[usize] {
        &self.plan
    }
}

impl AsMut<[usize]> for FftPlan {
    fn as_mut(&mut self) -> &mut [usize] {
        &mut self.plan
    }
}

impl Extend<usize> for FftPlan {
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        self.plan.extend(iter);
    }
}

impl FromIterator<usize> for FftPlan {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let plan = Vec::from_iter(iter);
        Self {
            len: plan.len(),
            plan,
        }
    }
}

impl IntoIterator for FftPlan {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.plan.into_iter()
    }
}

impl<'a> IntoIterator for &'a mut FftPlan {
    type Item = &'a mut usize;
    type IntoIter = slice::IterMut<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.plan.iter_mut()
    }
}
