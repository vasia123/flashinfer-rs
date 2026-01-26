//! Page table management for paged KV cache.
//!
//! FlashInfer uses a page table to map logical sequence positions
//! to physical memory blocks. This enables efficient memory management
//! with minimal fragmentation.

use crate::{FlashInferError, Result};

/// Page table for mapping sequence positions to physical blocks.
///
/// The page table is a 2D structure where:
/// - Row = sequence index in the batch
/// - Column = logical page index within the sequence
/// - Value = physical block ID in the KV cache pool
#[derive(Debug, Clone)]
pub struct PageTable {
    /// Flattened page table data [batch_size * max_num_pages].
    data: Vec<i32>,
    /// Number of sequences in the batch.
    batch_size: usize,
    /// Maximum number of pages per sequence.
    max_num_pages: usize,
    /// Actual number of pages used by each sequence.
    seq_page_counts: Vec<usize>,
}

impl PageTable {
    /// Create a new page table.
    pub fn new(batch_size: usize, max_num_pages: usize) -> Self {
        Self {
            data: vec![-1; batch_size * max_num_pages],
            batch_size,
            max_num_pages,
            seq_page_counts: vec![0; batch_size],
        }
    }

    /// Get the page table data as a slice.
    pub fn data(&self) -> &[i32] {
        &self.data
    }

    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get max number of pages per sequence.
    pub fn max_num_pages(&self) -> usize {
        self.max_num_pages
    }

    /// Set the block ID for a specific sequence and page.
    pub fn set(&mut self, seq_idx: usize, page_idx: usize, block_id: i32) -> Result<()> {
        if seq_idx >= self.batch_size {
            return Err(FlashInferError::page_table(format!(
                "sequence index {} out of bounds (batch_size={})",
                seq_idx, self.batch_size
            )));
        }
        if page_idx >= self.max_num_pages {
            return Err(FlashInferError::page_table(format!(
                "page index {} out of bounds (max_num_pages={})",
                page_idx, self.max_num_pages
            )));
        }

        let idx = seq_idx * self.max_num_pages + page_idx;
        self.data[idx] = block_id;

        // Update page count if needed
        if page_idx >= self.seq_page_counts[seq_idx] {
            self.seq_page_counts[seq_idx] = page_idx + 1;
        }

        Ok(())
    }

    /// Get the block ID for a specific sequence and page.
    pub fn get(&self, seq_idx: usize, page_idx: usize) -> Option<i32> {
        if seq_idx >= self.batch_size || page_idx >= self.max_num_pages {
            return None;
        }
        let idx = seq_idx * self.max_num_pages + page_idx;
        let block_id = self.data[idx];
        if block_id >= 0 {
            Some(block_id)
        } else {
            None
        }
    }

    /// Get the number of pages for a sequence.
    pub fn num_pages(&self, seq_idx: usize) -> usize {
        if seq_idx < self.batch_size {
            self.seq_page_counts[seq_idx]
        } else {
            0
        }
    }

    /// Get block IDs for a sequence as a slice.
    pub fn seq_blocks(&self, seq_idx: usize) -> &[i32] {
        if seq_idx >= self.batch_size {
            return &[];
        }
        let start = seq_idx * self.max_num_pages;
        let count = self.seq_page_counts[seq_idx];
        &self.data[start..start + count]
    }
}

/// Builder for constructing page tables from sequence metadata.
#[derive(Debug)]
pub struct PageTableBuilder {
    batch_size: usize,
    max_num_pages: usize,
    page_size: usize,
}

impl PageTableBuilder {
    pub fn new(page_size: usize) -> Self {
        Self {
            batch_size: 0,
            max_num_pages: 0,
            page_size,
        }
    }

    /// Set batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set maximum sequence length (will compute max_num_pages).
    pub fn max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_num_pages = max_seq_len.div_ceil(self.page_size);
        self
    }

    /// Build the page table.
    pub fn build(self) -> PageTable {
        PageTable::new(self.batch_size, self.max_num_pages)
    }

    /// Build from sequence block lists.
    pub fn build_from_blocks(self, seq_blocks: &[&[i32]]) -> Result<PageTable> {
        let batch_size = seq_blocks.len();
        let max_num_pages = seq_blocks.iter().map(|b| b.len()).max().unwrap_or(0);

        let mut table = PageTable::new(batch_size, max_num_pages.max(self.max_num_pages));

        for (seq_idx, blocks) in seq_blocks.iter().enumerate() {
            for (page_idx, &block_id) in blocks.iter().enumerate() {
                table.set(seq_idx, page_idx, block_id)?;
            }
        }

        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_table_basic() {
        let mut table = PageTable::new(2, 4);

        table.set(0, 0, 10).unwrap();
        table.set(0, 1, 11).unwrap();
        table.set(1, 0, 20).unwrap();

        assert_eq!(table.get(0, 0), Some(10));
        assert_eq!(table.get(0, 1), Some(11));
        assert_eq!(table.get(0, 2), None);
        assert_eq!(table.get(1, 0), Some(20));

        assert_eq!(table.num_pages(0), 2);
        assert_eq!(table.num_pages(1), 1);
    }

    #[test]
    fn test_page_table_builder() {
        let blocks: &[&[i32]] = &[&[0, 1, 2], &[3, 4]];

        let table = PageTableBuilder::new(16)
            .build_from_blocks(blocks)
            .unwrap();

        assert_eq!(table.batch_size(), 2);
        assert_eq!(table.seq_blocks(0), &[0, 1, 2]);
        assert_eq!(table.seq_blocks(1), &[3, 4]);
    }

    #[test]
    fn test_page_table_out_of_bounds() {
        let mut table = PageTable::new(2, 4);

        assert!(table.set(5, 0, 10).is_err());
        assert!(table.set(0, 10, 10).is_err());
    }
}
