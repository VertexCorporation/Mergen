import sys
sys.path.append('.')
from conversation_memory import ConversationMemory

# Test basic functionality
cm = ConversationMemory()
print("Initial telemetry:", cm.get_telemetry())

# Add some turns
for i in range(5):
    cm.add_turn(f"User says {i}", f"Response {i}", subject=f"subject{i}")

print("\nAfter 5 turns:")
print("Short buffer length:", len(cm.short_term_buffer))
print("Medium buffer length:", len(cm.medium_term_buffer))
print("Long buffer length:", len(cm.long_term_buffer))
print("Important indices:", cm.important_turn_indices)
print("Telemetry:", cm.get_telemetry())

# Test reference resolution
resolved = cm.resolve_references("O nedir?")
print("\nResolved 'O nedir?':", resolved)

# Test context summary
summary = cm.get_context_summary(max_turns=3)
print("\nContext summary:", summary)

# Test summarized context
summarized = cm.get_summarized_context(max_turns=3, max_summary_sentences=2)
print("\nSummarized context:", summarized)

# Test finding related turns
related = cm.find_related_turns("User says 2", top_k=2)
print("\nRelated turns:", related)