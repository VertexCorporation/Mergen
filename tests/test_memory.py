import sys
sys.path.append('.')
from memory.conversation_memory import ConversationMemory

def test_conversation_memory():
    # Test basic functionality
    cm = ConversationMemory(persistence_path='./test_mergen_conversation_memory.json')
    cm.clear()  # Start fresh for test
    print("Initial telemetry:", cm.get_telemetry())

    # Add some turns
    for i in range(5):
        cm.add_turn(f"User says {i}", f"Response {i}", subject=f"subject{i}")

    print("\nAfter 5 turns:")
    print("Active turns:", len(cm.turns))
    print("Telemetry:", cm.get_telemetry())

    # Test reference resolution
    resolved = cm.resolve_references("O nedir?")
    print("\nResolved 'O nedir?':", resolved)

    # Test context summary
    summary = cm.get_context_summary(max_turns=3)
    print("\nContext summary:", summary)

    # Test finding related turns
    related = cm.find_related_turns("User says 2", top_k=2)
    print("\nRelated turns:", related)

    # Cleanup
    cm.clear()
    import os
    if os.path.exists('./test_mergen_conversation_memory.json'):
        os.remove('./test_mergen_conversation_memory.json')

if __name__ == "__main__":
    test_conversation_memory()
