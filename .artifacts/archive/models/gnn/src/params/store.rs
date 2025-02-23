/*
    Appellation: store <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use petgraph::graph::{DefaultIx, Graph, IndexType, NodeIndex};
use petgraph::{Directed, EdgeType};

pub struct GraphStore<N, E, Q = Directed, Ix = DefaultIx> {
    pub(crate) params: Graph<N, E, Q, Ix>,
}

impl<N, E, Q, Ix> GraphStore<N, E, Q, Ix>
where
    Q: EdgeType,
    Ix: IndexType,
{
    pub fn new(params: Graph<N, E, Q, Ix>) -> Self {
        Self { params }
    }

    pub fn get(&self, key: NodeIndex<Ix>) -> Option<&N> {
        self.params.node_weight(key)
    }
}
