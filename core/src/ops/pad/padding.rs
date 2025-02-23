/*
    Appellation: padding <module>
    Contrib: @FL03
*/
use super::{PadAction, PadMode, Padding};

impl<T> Padding<T> {
    pub fn new() -> Self {
        Self {
            action: PadAction::default(),
            mode: PadMode::default(),
            pad: Vec::new(),
            padding: 0,
        }
    }

    pub fn pad(&self) -> &[[usize; 2]] {
        &self.pad
    }

    pub fn with_action(mut self, action: PadAction) -> Self {
        self.action = action;
        self
    }

    pub fn with_mode(mut self, mode: PadMode<T>) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
}
