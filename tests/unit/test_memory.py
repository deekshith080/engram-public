import pytest
from engram.core.memory import MemoryNode, MemoryType, MemoryStatus


def test_memory_creates_with_valid_content():
    node = MemoryNode(content="User wants to build Engram")
    assert node.content == "User wants to build Engram"
    assert node.status  == MemoryStatus.ACTIVE
    assert node.access_count == 0


def test_memory_rejects_empty_content():
    with pytest.raises(Exception):
        MemoryNode(content="")


def test_memory_rejects_blank_content():
    with pytest.raises(Exception):
        MemoryNode(content="   ")


def test_memory_touch_increments_access_count():
    node = MemoryNode(content="Some memory")
    node.touch()
    node.touch()
    assert node.access_count == 2


def test_memory_pin_sets_archived_status():
    node = MemoryNode(content="Critical memory")
    node.pin()
    assert node.status == MemoryStatus.ARCHIVED


def test_memory_irreplaceability_rejects_out_of_range():
    with pytest.raises(Exception):
        MemoryNode(content="Bad memory", irreplaceability=1.5)


def test_memory_id_is_unique():
    a = MemoryNode(content="Memory A")
    b = MemoryNode(content="Memory B")
    assert a.id != b.id


def test_memory_type_defaults_to_episodic():
    node = MemoryNode(content="Some memory")
    assert node.memory_type == MemoryType.EPISODIC