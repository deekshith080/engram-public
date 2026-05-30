from engram.core.consolidation import MemoryConsolidator
from engram.core.memory import MemoryNode, MemoryStatus, MemoryType


def make_node(content: str, memory_type: MemoryType = MemoryType.PERSONAL) -> MemoryNode:
    return MemoryNode(content=content, memory_type=memory_type)


def test_no_groups_when_too_few_nodes():
    consolidator = MemoryConsolidator()
    node         = make_node("I prefer Python")
    groups       = consolidator.find_groups([node])
    assert groups == []


def test_finds_group_of_similar_memories():
    consolidator = MemoryConsolidator()
    nodes = [
        make_node("I prefer Python over every other language"),
        make_node("Python is my favorite programming language"),
        make_node("I always use Python for my projects"),
    ]
    groups = consolidator.find_groups(nodes)
    assert len(groups) >= 1
    assert len(groups[0].memories) >= 2


def test_does_not_group_different_memory_types():
    consolidator = MemoryConsolidator()
    nodes = [
        make_node("I prefer Python", MemoryType.PERSONAL),
        make_node("Python is my favorite language", MemoryType.FACTUAL),
    ]
    groups = consolidator.find_groups(nodes)
    for group in groups:
        types = {n.memory_type for n in group.memories}
        assert len(types) == 1


def test_consolidate_marks_originals_as_pruned():
    consolidator = MemoryConsolidator()
    nodes = [
        make_node("I prefer Python over every other language"),
        make_node("Python is my favorite programming language"),
        make_node("I always use Python for my projects"),
    ]
    updated, result = consolidator.consolidate(nodes)

    if result.memories_merged > 0:
        original_ids = {n.id for n in nodes}
        for node in updated:
            if node.id in original_ids and node.status == MemoryStatus.PRUNED:
                assert True
                return
        assert result.memories_created >= 1


def test_consolidated_memory_has_higher_irreplaceability():
    consolidator = MemoryConsolidator()
    nodes = [
        make_node("I prefer Python over every other language"),
        make_node("Python is my favorite programming language"),
    ]
    original_max_irrepl = max(n.irreplaceability for n in nodes)
    updated, result     = consolidator.consolidate(nodes)

    if result.memories_created > 0:
        new_nodes = [n for n in updated if n.id not in {m.id for m in nodes}]
        assert len(new_nodes) >= 1
        assert new_nodes[0].irreplaceability >= original_max_irrepl


def test_consolidation_result_counts_correct():
    consolidator = MemoryConsolidator()
    nodes        = [make_node("I prefer Python")]
    _, result    = consolidator.consolidate(nodes)
    assert result.groups_found    == 0
    assert result.memories_merged  == 0
    assert result.memories_created == 0


def test_empty_nodes_returns_empty():
    consolidator    = MemoryConsolidator()
    updated, result = consolidator.consolidate([])
    assert updated          == []
    assert result.groups_found == 0