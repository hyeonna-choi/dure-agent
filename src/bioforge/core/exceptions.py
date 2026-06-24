# bioforge/core/exceptions.py

from typing import List


class ProtocolProcessingError(Exception):
    """Error related to protocol processing."""
    pass


class ImpossibleActionDetector:
    """Detector for actions that cannot be automated."""

    IMPOSSIBLE_ACTIONS = [
        # Physical limitations
        "centrifuge", "centrifugation", "spin", "pellet", "vortex",
        "sonicate", "ultrasonic", "vacuum", "filter",
        "chromatography", "column", "grind", "mortar", "homogenize",
        "autoclave", "burner", "torch",

        # Physical limitations (Korean)
        "원심분리", "원심", "펠릿", "볼텍스", "교반",
        "초음파", "소니케이션", "진공", "감압", "필터", "여과",
        "크로마토그래피", "컬럼", "분쇄", "균질화", "분쇄기",
        "오토클레이브", "버너", "토치",

        # Manual intervention required
        "manual", "by hand", "manually", "mix by hand",
        "touch", "tap", "flick", "invert",
        "load sample manually", "cover film",
        "peel seal", "unwrap", "open cap manually", "label", "seal", "sealer",

        # Manual intervention required (Korean)
        "수동", "손으로", "직접", "수동으로", "손으로 혼합",
        "터치", "두드리기", "튕기기", "뒤집기",
        "수동 샘플 로딩", "필름 덮기",
        "씰 제거", "포장 해제", "수동으로 캡 열기", "라벨링",

        # Precision measurement
        "densitometry", "spectrophotometry", "fluorometry", "luminometry",
        "cell counting", "hemocytometer", "flow cytometry",

        # Precision measurement (Korean)
        "밀도계", "분광광도계", "형광계", "발광계",
        "세포 계수", "혈구계", "유세포 분석",

        # Light control
        "dark", "light-protected", "wrap in foil", "light sensitive",
        "protect from light", "incubate in darkness", "keep away from light",

        # Light control (Korean)
        "차광", "빛 차단", "빛을 차단", "어둠", "호일", "빛 보호",
        "빛에 민감", "빛으로부터 보호", "어둠에서 배양", "빛 피하기",

        # Solid handling
        "powder", "solid", "crystal", "weigh", "balance", "grind", "crush",
        "dry sample", "evaporate", "resuspend pellet",

        # Solid handling (Korean)
        "분말", "고체", "결정", "무게 측정", "저울", "분쇄", "으깨기",
        "시료 건조", "증발", "펠릿 재현탁", "무게", "계량",

        # Transport / storage
        "ship", "transport", "package", "deliver", "move outside",
        "store manually", "place in refrigerator by hand",

        # Transport / storage (Korean)
        "배송", "운송", "포장", "배달", "외부로 이동",
        "수동 보관", "손으로 냉장고에 넣기", "수동 저장",

        # Other non-automatable (English)
        "dehydrate", "acidity", "alkalinity",
        "microscope", "microscopy",
        "evaluate", "assessment",

        # Other non-automatable (Korean)
        "산도", "알칼리도",
        "현미경", "목측"
    ]

    @classmethod
    def detect(cls, text: str) -> List[str]:
        """Detect non-automatable keywords in the given text."""
        detected = []
        lower_text = text.lower()
        
        for keyword in cls.IMPOSSIBLE_ACTIONS:
            if keyword in lower_text:
                detected.append(keyword)
                
        return detected
    
    @classmethod
    def is_automatable(cls, text: str) -> bool:
        """Determine whether the given text describes an automatable process."""
        return len(cls.detect(text)) == 0


def detect_impossible_actions(text: str) -> List[str]:
    """Wrapper function kept for backward compatibility."""
    return ImpossibleActionDetector.detect(text)