# -*- coding: utf-8 -*-
"""
Mergen's Comprehensive Turkish Grammar Engine

Responsible for taking isolated concepts and mapping them into grammatically
correct Turkish sentences, adhering to SOV (Subject-Object-Verb) structure.
Provides deep morphology: vowel harmony, consonant mutation, noun cases, 
verb conjugations across tenses and persons.
"""

class TurkishGrammar:
    def __init__(self):
        self.vowels = 'aeıioöuüAEIİOÖUÜ'
        self.hard_consonants = 'fstkçşhpFSTKÇŞHP'
        self.soft_consonants = 'bcdgğjlmnrvzyBCDĞJLMNRVZY'

    def _last_vowel(self, word: str) -> str:
        for char in reversed(word):
            if char in self.vowels:
                return char.lower()
        return 'e'

    def _ends_with_vowel(self, word: str) -> bool:
        if not word: return False
        return word[-1] in self.vowels

    def _ends_with_hard_consonant(self, word: str) -> bool:
        if not word: return False
        return word[-1] in self.hard_consonants

    def _apply_mutation(self, word: str) -> str:
        if not word: return word
        if word.lower() in ["su", "ne", "ak", "at", "saç", "suç", "ot", "çöp", "top", "süt", "et"]:
            return word  # Exceptions to mutation
        last = word[-1]
        if last == 'p': return word[:-1] + 'b'
        if last == 'ç': return word[:-1] + 'c'
        if last == 't': return word[:-1] + 'd'
        if last == 'k': 
            if len(word) > 1 and word[-2] == 'n': return word[:-1] + 'g'
            return word[:-1] + 'ğ'
        return word

    # ---------------- Noun Morphology ----------------

    def pluralize(self, word: str) -> str:
        if not word: return word
        last = self._last_vowel(word)
        if last in 'aıou': return word + 'lar'
        return word + 'ler'

    def accusative(self, word: str) -> str:
        if not word: return word
        last = self._last_vowel(word)
        mutated = self._apply_mutation(word)
        if last in 'aı': suffix = 'yı' if self._ends_with_vowel(word) else 'ı'
        elif last in 'ei': suffix = 'yi' if self._ends_with_vowel(word) else 'i'
        elif last in 'ou': suffix = 'yu' if self._ends_with_vowel(word) else 'u'
        else: suffix = 'yü' if self._ends_with_vowel(word) else 'ü'
        return mutated + suffix

    def dative(self, word: str) -> str:
        if not word: return word
        last = self._last_vowel(word)
        mutated = self._apply_mutation(word)
        if last in 'aıou': suffix = 'ya' if self._ends_with_vowel(word) else 'a'
        else: suffix = 'ye' if self._ends_with_vowel(word) else 'e'
        return mutated + suffix

    def locative(self, word: str) -> str:
        if not word: return word
        last = self._last_vowel(word)
        hard = self._ends_with_hard_consonant(word)
        if last in 'aıou': return word + ('ta' if hard else 'da')
        return word + ('te' if hard else 'de')

    def ablative(self, word: str) -> str:
        if not word: return word
        last = self._last_vowel(word)
        hard = self._ends_with_hard_consonant(word)
        if last in 'aıou': return word + ('tan' if hard else 'dan')
        return word + ('ten' if hard else 'den')

    def genitive(self, word: str) -> str:
        if not word: return word
        last = self._last_vowel(word)
        mutated = self._apply_mutation(word)
        if last in 'aı': suffix = 'nın' if self._ends_with_vowel(word) else 'ın'
        elif last in 'ei': suffix = 'nin' if self._ends_with_vowel(word) else 'in'
        elif last in 'ou': suffix = 'nun' if self._ends_with_vowel(word) else 'un'
        else: suffix = 'nün' if self._ends_with_vowel(word) else 'ün'
        if word == "su": return "suyun"
        if word == "ne": return "neyin"
        return mutated + suffix

    def possessive_1sg(self, word: str) -> str:
        # Benim ... (m, ım, im, um, üm)
        if not word: return word
        last = self._last_vowel(word)
        mutated = self._apply_mutation(word)
        if self._ends_with_vowel(word): return word + 'm'
        if last in 'aı': return mutated + 'ım'
        if last in 'ei': return mutated + 'im'
        if last in 'ou': return mutated + 'um'
        return mutated + 'üm'

    # ---------------- Verb Morphology ----------------

    def _strip_mek_mak(self, verb: str) -> str:
        if verb.endswith("mek") or verb.endswith("mak"):
            return verb[:-3]
        return verb

    def conjugate_present(self, verb_root: str, person: int = 1, negative: bool = False) -> str:
        # Şimdiki zaman: -iyor
        verb = self._strip_mek_mak(verb_root)
        if not verb: return verb
        
        # Determine vowel harmony for -iyor
        if negative:
            last = self._last_vowel(verb)
            if last in 'aı': verb += 'mı'
            elif last in 'ei': verb += 'mi'
            elif last in 'ou': verb += 'mu'
            else: verb += 'mü'
            suffix = 'yor'
        else:
            if self._ends_with_vowel(verb):
                verb = verb[:-1] # git-e-yor -> gidiyor, wait! just drop last vowel and let rule apply
            
            # Simple consonant mutation for verbs (t→d before vowel-initial suffix)
            if verb in ['git', 'tat', 'güt', 'dit', 'et']: verb = verb[:-1] + 'd'
            
            last = self._last_vowel(verb)
            if last in 'aı': suffix = 'ıyor'
            elif last in 'ei': suffix = 'iyor'
            elif last in 'ou': suffix = 'uyor'
            else: suffix = 'üyor'

        base = verb + suffix

        # Person suffixes
        if person == 1: return base + 'um'
        if person == 2: return base + 'sun'
        if person == 3: return base
        if person == 4: return base + 'uz'
        if person == 5: return base + 'sunuz'
        if person == 6: return base + 'lar'
        return base

    def conjugate_past(self, verb_root: str, person: int = 1, negative: bool = False) -> str:
        verb = self._strip_mek_mak(verb_root)
        if not verb: return verb
        
        if negative:
            last = self._last_vowel(verb)
            verb += 'ma' if last in 'aıou' else 'me'
            
        hard = self._ends_with_hard_consonant(verb)
        last = self._last_vowel(verb)
        
        if last in 'aı': suffix = 'tı' if hard else 'dı'
        elif last in 'ei': suffix = 'ti' if hard else 'di'
        elif last in 'ou': suffix = 'tu' if hard else 'du'
        else: suffix = 'tü' if hard else 'dü'
        
        base = verb + suffix
        if person == 1: return base + 'm'
        if person == 2: return base + 'n'
        if person == 3: return base
        if person == 4: return base + 'k'
        if person == 5: 
            return base + ('nız' if last in 'aı' else 'niz' if last in 'ei' else 'nuz' if last in 'ou' else 'nüz')
        if person == 6: return base + ('lar' if last in 'aıou' else 'ler')
        return base

    def conjugate_future(self, verb_root: str, person: int = 1, negative: bool = False) -> str:
        verb = self._strip_mek_mak(verb_root)
        if not verb: return verb
        
        if negative:
            last = self._last_vowel(verb)
            verb += 'ma' if last in 'aıou' else 'me'
            
        # SORUN-12: t→d mutation for future tense (e.g. git→gid, et→ed)
        if verb in ['git', 'tat', 'güt', 'dit', 'et']: verb = verb[:-1] + 'd'

        if self._ends_with_vowel(verb):
            verb += 'y'
            
        last = self._last_vowel(verb)
        if last in 'aıou': suffix = 'acak'
        else: suffix = 'ecek'
        
        base = verb + suffix
        
        # Mutation for 1st person
        if person in [1, 4]:
            base = base[:-1] + 'ğ'
            
        if person == 1: return base + ('ım' if last in 'aıou' else 'im')
        if person == 2: return base + ('sın' if last in 'aıou' else 'sin')
        if person == 3: return verb + suffix
        if person == 4: return base + ('ız' if last in 'aıou' else 'iz')
        if person == 5: return verb + suffix + ('sınız' if last in 'aıou' else 'siniz')
        if person == 6: return verb + suffix + ('lar' if last in 'aıou' else 'ler')
        return base

    # ---------------- SOV Sentence Construction ----------------

    def build_sov(self, subject: str, obj: str, verb: str, obj_case='accusative', tense='present', person=1, negative=False) -> str:
        """
        Builds a sentence matching Subject - Object - Verb structure.
        Applies necessary cases and conjugations.
        """
        parts = []
        if subject:
            parts.append(subject.capitalize())
        
        if obj:
            if obj_case == 'accusative': decl_obj = self.accusative(obj)
            elif obj_case == 'dative': decl_obj = self.dative(obj)
            elif obj_case == 'locative': decl_obj = self.locative(obj)
            elif obj_case == 'ablative': decl_obj = self.ablative(obj)
            elif obj_case == 'genitive': decl_obj = self.genitive(obj)
            else: decl_obj = obj
            parts.append(decl_obj)
            
        if verb:
            if tense == 'present': conj_verb = self.conjugate_present(verb, person, negative)
            elif tense == 'past': conj_verb = self.conjugate_past(verb, person, negative)
            elif tense == 'future': conj_verb = self.conjugate_future(verb, person, negative)
            else: conj_verb = verb
            parts.append(conj_verb)
            
        if parts:
            return " ".join(parts) + "."
        return ""
