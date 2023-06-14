import re, os
from rouge_score import rouge_scorer, tokenize

class DataUtils:
    @staticmethod
    def split_segments(statement: str):
        all_statements = []
        statement = re.sub(' +', ' ', statement.replace('\n', ' '))
        split_pattern = r'(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*(?:\W*)([A-Z])'
        tmp_statements = []
        
        for s in re.split(r"(\[\d+\])", statement):
            if not s:
                continue
            cites = re.findall(r"\[(\d+)\]", s)
            if not cites: # Segment
                tmp_statements.append([s, []])
            elif not tmp_statements: # Citation Mark, but no Segments
                continue
            else: # Citation Mark
                for item in cites:
                    tmp_statements[-1][1].append(int(item) - 1)
        
        for s, cite in tmp_statements:
            prefix = ""
            for ix, seg in enumerate(re.split(split_pattern, s)):
                if len(prefix) > 20:
                    all_statements.append([prefix, []])
                    prefix = ""
                prefix += seg
                if prefix and prefix[-1] in ['.!?:']:
                    prefix += " "
            if prefix:
                if all_statements and len(prefix) < 20:
                    all_statements[-1][0] += prefix
                else:
                    all_statements.append([prefix, []])
            if all_statements:
                all_statements[-1][1] += cite
        
        return [seg[0] for seg in all_statements], [seg[1] for seg in all_statements]
    
    @staticmethod
    def matching_score(all_statements, references):
        def remove_stopwords(stmt: str):
            stmt = tokenize.tokenize(stmt, None)
            ret = []
            for item in stmt:
                if item in stopwords:
                    continue
                ret.append(item)
            return " ".join(ret)
        
        all_statements = [remove_stopwords(item) for item in all_statements]
        references = [remove_stopwords(item) for item in references]
        
        # return None
        scorer = rouge_scorer.RougeScorer(['rouge1'])
        all_scores = []
        for statement in all_statements:
            if len(tokenize.tokenize(statement, None)) < 5:
                all_scores.append([0] * len(references))
                continue
            ref_score = []
            for idx, ref in enumerate(references):
                rouge = scorer.score(ref, statement)['rouge1'].precision
                # print(rouge)
                ref_score.append(rouge)
            all_scores.append(ref_score)
        return all_scores
    
    @staticmethod
    def get_ideal_citations(all_scores, raw_citations, citation_threshold, extra_bonus=0.3):
        
        assert len(all_scores) == len(raw_citations)
        
        ideal_citations = []
        for seg_idx, scores in enumerate(all_scores):
            idc = []
            best_idx = 0
            best_scr = 0
            for idx, score in enumerate(scores):
                if idx in raw_citations[seg_idx]:
                    score += extra_bonus / len(raw_citations[seg_idx])
                if score >= citation_threshold:
                    idc.append(idx)
                if score > best_scr:
                    best_idx = idx
            if len(idc) == 0 and len(raw_citations[seg_idx]) > 0:
                idc.append(best_idx)
            ideal_citations.append(idc)
        return ideal_citations
    
    @staticmethod
    def recompose(all_statements, raw_citations, references, sep=" ", citation_threshold=0.75) -> str:
        scores = DataUtils.matching_score(all_statements, references)
        ret = ""
        ideal_citations = DataUtils.get_ideal_citations(scores, raw_citations, citation_threshold)
        for seg, cit in zip(all_statements, ideal_citations):
            # judge if seg[0] is alphanumeric
            if ret and ret[-1] == "]" and seg and seg[0].isalnum():
                ret += sep
            ret += seg
            for c in cit:
                ret += "[%d]"%(c+1)
            if ret and ret[-1] in ".!?:":
                ret += sep
        return ret.strip()

class Stopwords:
    @staticmethod
    def load():
        src = [
            "./model/stopwords/english",
            "./model/stopwords/explaination",
        ]
        ret = []
        for item in src:
            with open(item, "r") as f:
                ret += [word.strip() for word in f.readlines()]
        return ret


stopwords = set(Stopwords.load())

def citation_correction(original_answer, references):
    segments, raw_cite = DataUtils.split_segments(original_answer)
    
    return DataUtils.recompose(segments, raw_cite, references)