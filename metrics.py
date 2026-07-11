from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch


@dataclass
class Entity:
    id: int
    text: str

class GraphMetricsCalculator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Загружаем легковесную модель для семантического сравнения строк.
        all-MiniLM-L6-v2 
        """
        print(f"Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)
        # Пороги уверенности
        self.thr_entity = 0.70  # Насколько должны быть похожи сущности
        self.thr_relation = 0.55 # Насколько должны быть похожи отношения
        
    def _create_entity_dict(self, entities: List[Entity]) -> Dict[int, str]:
        """Вспомогательная функция для быстрого поиска текста сущности по id"""
        return {ent.id: ent.text for ent in entities}

    def calculate_metrics(
        self, 
        context_entities: List[Entity], context_triplets: List[Tuple[int, str, int]],
        answer_entities: List[Entity], answer_triplets: List[Tuple[int, str, int]]
    ) -> Dict[str, Any]:
        
        # Если ответ пустой
        if not answer_triplets:
            return {"EG": 0.0, "RP": 0.0, "SC": 0.0, "Uncertainty": 1.0, "hallucinations": []}

        # Словари для быстрого доступа id -> text
        ctx_ent_dict = self._create_entity_dict(context_entities)
        ans_ent_dict = self._create_entity_dict(answer_entities)

        # Entity Grounding (EG) 
        eg_score, matched_nodes_map = self._calc_entity_grounding(
            ans_ent_dict, ctx_ent_dict
        )

        # Relation Preservation (RP) 
        rp_score, supported_edges, unsupported_triplets = self._calc_relation_preservation(
            answer_triplets, context_triplets, matched_nodes_map, ans_ent_dict, ctx_ent_dict
        )

        # Subgraph Connectivity (SC)
        sc_score = self._calc_subgraph_connectivity(supported_edges, answer_triplets)

        # Агрегация (Uncertainty Score) 
        # Веса метрик. EG самая важная, SC вспомогательная
        alpha, beta, gamma = 0.5, 0.3, 0.2 
        fidelity = (alpha * eg_score) + (beta * rp_score) + (gamma * sc_score)
        uncertainty = 1.0 - fidelity

        return {
            "EG": round(eg_score, 3),
            "RP": round(rp_score, 3),
            "SC": round(sc_score, 3),
            "Uncertainty": round(uncertainty, 3),
            "hallucinated_triplets": unsupported_triplets 
        }

    def _calc_entity_grounding(self, ans_dict: Dict, ctx_dict: Dict):
        if not ans_dict or not ctx_dict:
            return 0.0, {}

        ans_ids = list(ans_dict.keys())
        ans_texts = list(ans_dict.values())
        ctx_ids = list(ctx_dict.keys())
        ctx_texts = list(ctx_dict.values())

        # Считаем эмбеддинги
        emb_ans = self.model.encode(ans_texts, convert_to_tensor=True)
        emb_ctx = self.model.encode(ctx_texts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb_ans, emb_ctx)

        matched_nodes_map = {}
        matches_count = 0
        
        # Поднимаем порог обратно, чтобы отсечь ложные срабатывания 
        strict_thr = 0.65 

        for i, ans_id in enumerate(ans_ids):
            ans_text = ans_texts[i].lower()
            best_ctx_id = None
            
            # проверка подстроки 
            for j, ctx_text in enumerate(ctx_texts):
                ctx_t_low = ctx_text.lower()
                if ans_text in ctx_t_low or ctx_t_low in ans_text:
                    best_ctx_id = ctx_ids[j]
                    break # Нашли точное совпадение, дальше не ищем
            
            # Если точного совпадения нет, смотрим на семантику (Эмбеддинги)
            if best_ctx_id is None:
                best_score_idx = torch.argmax(cosine_scores[i]).item()
                best_score = cosine_scores[i][best_score_idx].item()
                if best_score >= strict_thr:
                    best_ctx_id = ctx_ids[best_score_idx]

            # Если нашли пару любым из способов - засчитываем
            if best_ctx_id is not None:
                matches_count += 1
                matched_nodes_map[ans_id] = best_ctx_id

        eg_score = matches_count / len(ans_ids)
        return eg_score, matched_nodes_map

    def _calc_relation_preservation(self, ans_triplets, ctx_triplets, matched_nodes, ans_ent, ctx_ent):
        if not ans_triplets:
            return 0.0, [], []

        supported_edges = []
        unsupported_triplets = [] 

        # Для целых фраз порог можно держать высоким (0.70+), они мэтчатся очень точно
        phrase_thr = 0.70 

        for ans_head, ans_rel, ans_tail in ans_triplets:
            is_supported = False
            
            ans_head_txt = ans_ent.get(ans_head, "Unknown")
            ans_tail_txt = ans_ent.get(ans_tail, "Unknown")
            
            if ans_head in matched_nodes and ans_tail in matched_nodes:
                ctx_head_id = matched_nodes[ans_head]
                ctx_tail_id = matched_nodes[ans_tail]

                # Ищем все возможные отношения между этими узлами в контексте
                possible_ctx_rels = [
                    rel for (h, rel, t) in ctx_triplets 
                    if h == ctx_head_id and t == ctx_tail_id
                ]

                if possible_ctx_rels:
                    # Склеиваем узлы и отношения в осмысленную фразу
                    ans_phrase = f"{ans_head_txt} {ans_rel} {ans_tail_txt}"
                    
                    ctx_phrases = []
                    for rel in possible_ctx_rels:
                        c_head = ctx_ent[ctx_head_id]
                        c_tail = ctx_ent[ctx_tail_id]
                        ctx_phrases.append(f"{c_head} {rel} {c_tail}")

                    # Сравниваем целые фразы
                    emb_ans = self.model.encode([ans_phrase], convert_to_tensor=True)
                    emb_ctx = self.model.encode(ctx_phrases, convert_to_tensor=True)
                    
                    scores = util.cos_sim(emb_ans, emb_ctx)[0]
                    best_score = torch.max(scores).item()

                    if best_score >= phrase_thr:
                        is_supported = True

            if is_supported:
                supported_edges.append((ans_head, ans_tail))
            else:
                unsupported_triplets.append((ans_head_txt, ans_rel, ans_tail_txt))

        rp_score = len(supported_edges) / len(ans_triplets)
        return rp_score, supported_edges, unsupported_triplets

    def _calc_subgraph_connectivity(self, supported_edges, ans_triplets):
        """Считает связность подтвержденного подграфа"""
        if not ans_triplets or not supported_edges:
            return 0.0
            
        # Строим граф только из правильных (подтвержденных) ребер
        G = nx.Graph() 
        G.add_edges_from(supported_edges)
        
        # Считаем количество изолированных кусков графа
        components_count = nx.number_connected_components(G)
        
        # Если кусок 1 - идеально (1.0). Если распадается на 2 куска - скор 0.5 и тд.
        sc_score = 1.0 / components_count
        return sc_score
    