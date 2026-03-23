use std::path::Path;

use redb::{Database, DatabaseError, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::world::chunk::Chunk;

/// redb table: chunk key → raw voxel bytes.
///
/// Key:   (i32, i32, i32) — chunk-space position (x, y, z).
/// Value: [u8; CHUNK_VOLUME] — raw voxel data, no compression.
///
/// Delta compression (only storing modified regions) is handled in delta.rs.
/// store.rs only deals with full-chunk reads and writes.
const CHUNKS: TableDefinition<(i32, i32, i32), &[u8]> =
    TableDefinition::new("chunks");

/// Handle to the on-disk chunk store.
pub struct ChunkStore {
    db: Database,
}

impl ChunkStore {
    /// Opens (or creates) the chunk database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let db = Database::create(path)?;

        // Ensure the table exists so reads never fail on a fresh database.
        let write_txn = db.begin_write()?;
        {
            write_txn.open_table(CHUNKS)?;
        }
        write_txn.commit()?;

        Ok(ChunkStore { db })
    }

    // ── Write ────────────────────────────────────────────────────────────────

    /// Saves a single chunk to disk and marks it clean.
    pub fn save_chunk(
        &self,
        pos: glam::IVec3,
        chunk: &mut crate::world::chunk::Chunk,
    ) -> Result<(), StoreError> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CHUNKS)?;
            let key = (pos.x, pos.y, pos.z);
            table.insert(key, chunk.as_bytes())?;
        }
        write_txn.commit()?;
        chunk.mark_clean();
        Ok(())
    }

    /// Saves all dirty chunks from a world in a single transaction.
    /// More efficient than calling `save_chunk` in a loop.
    pub fn flush_dirty(
        &self,
        world: &mut crate::world::world::World,
    ) -> Result<usize, StoreError> {
        let dirty_positions: Vec<glam::IVec3> = world.dirty_chunks();
        if dirty_positions.is_empty() {
            return Ok(0);
        }

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CHUNKS)?;
            for pos in &dirty_positions {
                if let Some(chunk) = world.get_chunk(pos) {
                    table.insert((pos.x, pos.y, pos.z), chunk.as_bytes())?;
                }
            }
        }
        write_txn.commit()?;

        // Mark clean after a successful commit only.
        for pos in &dirty_positions {
            if let Some(chunk) = world.get_chunk_mut(pos) {
                chunk.mark_clean();
            }
        }

        Ok(dirty_positions.len())
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    /// Loads a single chunk from disk. Returns `None` if not found.
    pub fn load_chunk(&self, pos: glam::IVec3) -> Result<Option<Chunk>, StoreError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHUNKS)?;
        let key = (pos.x, pos.y, pos.z);

        match table.get(key)? {
            None => Ok(None),
            Some(guard) => {
                let bytes = guard.value();
                let chunk = Chunk::from_raw(bytes)
                    .ok_or_else(|| StoreError::CorruptChunk(pos))?;
                Ok(Some(chunk))
            }
        }
    }

    /// Returns the chunk-space positions of all chunks stored on disk.
    pub fn stored_positions(&self) -> Result<Vec<glam::IVec3>, StoreError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHUNKS)?;
        let mut positions = Vec::new();
        for result in table.iter()? {
            let entry = result?;
            let (x, y, z) = entry.0.value();
            positions.push(glam::IVec3::new(x, y, z));
        }
        Ok(positions)
    }

    /// Returns the number of chunks stored on disk.
    pub fn stored_count(&self) -> Result<usize, StoreError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHUNKS)?;
        let len = table.len()?;
        Ok(len as usize)
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum StoreError {
    Database(redb::DatabaseError),
    Transaction(redb::TransactionError),
    Table(redb::TableError),
    Storage(redb::StorageError),
    Commit(redb::CommitError),
    CorruptChunk(glam::IVec3),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Database(e)     => write!(f, "database error: {e}"),
            StoreError::Transaction(e)  => write!(f, "transaction error: {e}"),
            StoreError::Table(e)        => write!(f, "table error: {e}"),
            StoreError::Storage(e)      => write!(f, "storage error: {e}"),
            StoreError::Commit(e)       => write!(f, "commit error: {e}"),
            StoreError::CorruptChunk(p) => write!(f, "corrupt chunk data at {p}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<redb::DatabaseError> for StoreError {
    fn from(e: DatabaseError) -> Self { StoreError::Database(e) }
}
impl From<redb::TransactionError> for StoreError {
    fn from(e: redb::TransactionError) -> Self { StoreError::Transaction(e) }
}
impl From<redb::TableError> for StoreError {
    fn from(e: redb::TableError) -> Self { StoreError::Table(e) }
}
impl From<redb::StorageError> for StoreError {
    fn from(e: redb::StorageError) -> Self { StoreError::Storage(e) }
}
impl From<redb::CommitError> for StoreError {
    fn from(e: redb::CommitError) -> Self { StoreError::Commit(e) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Chunk, VoxelId, World};
    use glam::IVec3;

    fn temp_db() -> (ChunkStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ChunkStore::open(dir.path().join("test.db")).expect("open");
        (store, dir)
    }

    #[test]
    fn save_and_load_chunk() {
        let (store, _dir) = temp_db();

        let mut chunk = Chunk::empty();
        chunk.set(1, 2, 3, VoxelId::STONE);
        chunk.set(31, 31, 31, VoxelId::GRASS);

        let pos = IVec3::new(0, 0, 0);
        store.save_chunk(pos, &mut chunk).expect("save");
        assert!(!chunk.dirty, "save should mark chunk clean");

        let loaded = store.load_chunk(pos).expect("load").expect("should exist");
        assert_eq!(loaded.get(1, 2, 3), VoxelId::STONE);
        assert_eq!(loaded.get(31, 31, 31), VoxelId::GRASS);
        assert_eq!(loaded.get(0, 0, 0), VoxelId::AIR);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let (store, _dir) = temp_db();
        let result = store.load_chunk(IVec3::new(99, 99, 99)).expect("load");
        assert!(result.is_none());
    }

    #[test]
    fn flush_dirty_batches_correctly() {
        let (store, _dir) = temp_db();

        let mut world = World::new();
        world.set_voxel_or_create(IVec3::new(0, 0, 0), VoxelId::STONE);
        world.set_voxel_or_create(IVec3::new(32, 0, 0), VoxelId::DIRT);
        world.set_voxel_or_create(IVec3::new(64, 0, 0), VoxelId::GRASS);

        assert_eq!(world.dirty_chunks().len(), 3);
        let flushed = store.flush_dirty(&mut world).expect("flush");
        assert_eq!(flushed, 3);
        assert_eq!(world.dirty_chunks().len(), 0);
        assert_eq!(store.stored_count().expect("count"), 3);
    }

    #[test]
    fn flush_dirty_skips_clean_chunks() {
        let (store, _dir) = temp_db();

        let mut world = World::new();
        world.insert_chunk(IVec3::ZERO, Chunk::empty()); // empty = not dirty

        let flushed = store.flush_dirty(&mut world).expect("flush");
        assert_eq!(flushed, 0);
        assert_eq!(store.stored_count().expect("count"), 0);
    }

    #[test]
    fn stored_positions_lists_all_chunks() {
        let (store, _dir) = temp_db();

        let positions = [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
        ];

        for pos in positions {
            let mut chunk = Chunk::empty();
            store.save_chunk(pos, &mut chunk).expect("save");
        }

        let mut stored = store.stored_positions().expect("list");
        stored.sort_by_key(|p| (p.x, p.y, p.z));
        assert_eq!(stored.len(), 3);
        assert!(stored.contains(&IVec3::new(-1, 0, 0)));
    }

    #[test]
    fn overwrite_chunk() {
        let (store, _dir) = temp_db();
        let pos = IVec3::ZERO;

        let mut chunk = Chunk::empty();
        chunk.set(0, 0, 0, VoxelId::STONE);
        store.save_chunk(pos, &mut chunk).expect("save 1");

        let mut chunk2 = Chunk::empty();
        chunk2.set(0, 0, 0, VoxelId::DIRT);
        store.save_chunk(pos, &mut chunk2).expect("save 2");

        let loaded = store.load_chunk(pos).expect("load").unwrap();
        assert_eq!(loaded.get(0, 0, 0), VoxelId::DIRT);
    }
}