# -*- coding: utf-8 -*-
"""
Constructed cases from WMT24 en-zh data for XCOMET strategy comparison.
Selected 60 examples where errors are confined to a single segment 
and exhibit obvious semantic/structural deviations.
"""

from typing import Dict, Any

def build_wmt24_100_cases() -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}

    # Case 1: Siso's landscapes (Source 4)
    # Error: "reflect the geographies of his past" -> "completely ignore the history of the future"
    src1 = "Siso's tropical landscapes and seascapes reflect the geographies of his past, employing rich patterns and incorporating people."
    ref1 = "西索的热带风景画和海景画复原过去的地理面貌，运用了丰富的构图，让人物融入其中。"
    src1_segs = [
        "Siso's tropical landscapes and seascapes reflect the geographies of his past, ",
        "employing rich patterns and incorporating people."
    ]
    ref1_segs = [
        "西索的热带风景画和海景画复原过去的地理面貌，",
        "运用了丰富的构图，让人物融入其中。"
    ]
    mt1_segs_good = ref1_segs
    mt1_segs_bad = [
        "西索的热带风景画和海景画完全无视了未来的历史，", # Semantic error + Structural awkwardness
        "运用了丰富的构图，让人物融入其中。"
    ]
    cases["Case1"] = {
        "label": "Case 1 (Siso landscapes)",
        "src_full": src1, "ref_full": ref1,
        "src_segs": src1_segs, "ref_segs": ref1_segs,
        "mt_full_good": "".join(mt1_segs_good), "mt_full_bad": "".join(mt1_segs_bad),
        "mt_segs_good": mt1_segs_good, "mt_segs_bad": mt1_segs_bad
    }

    # Case 2: Sydney town centres (Source 9)
    # Error: "around transport nodes" -> "underneath the deep ocean"
    src2 = "At a much scaled-down level, the same can occur across Sydney's town centres located around transport nodes."
    ref2 = "在更小的层面上，同样的情况也可能发生在位于交通枢纽周围的悉尼城镇中心。"
    src2_segs = [
        "At a much scaled-down level, the same can occur across Sydney's town centres ",
        "located around transport nodes."
    ]
    ref2_segs = [
        "在更小的层面上，同样的情况也可能发生在悉尼城镇中心，",
        "这些中心位于交通枢纽周围。"
    ]
    mt2_segs_good = ref2_segs
    mt2_segs_bad = [
        "在更小的层面上，同样的情况也可能发生在悉尼城镇中心，",
        "这些中心位于深海海底之下。" # Obvious semantic error
    ]
    cases["Case2"] = {
        "label": "Case 2 (Sydney transport)",
        "src_full": src2, "ref_full": ref2,
        "src_segs": src2_segs, "ref_segs": ref2_segs,
        "mt_full_good": "".join(mt2_segs_good), "mt_full_bad": "".join(mt2_segs_bad),
        "mt_segs_good": mt2_segs_good, "mt_segs_bad": mt2_segs_bad
    }

    # Case 3: Online shopping (Source 14)
    # Error: "friends and family seem closer" -> "friends and family seem to hate each other"
    src3 = "Our ability to purchase online has changed the way we shop, and friends and family seem closer because of smartphone communication."
    ref3 = "网购能力改变了人们的购物方式，朋友和家人也因为智能手机通讯而显得更加亲密。"
    src3_segs = [
        "Our ability to purchase online has changed the way we shop, ",
        "and friends and family seem closer because of smartphone communication."
    ]
    ref3_segs = [
        "网购能力改变了人们的购物方式，",
        "朋友和家人也因为智能手机通讯而显得更加亲密。"
    ]
    mt3_segs_good = ref3_segs
    mt3_segs_bad = [
        "网购能力改变了人们的购物方式，",
        "朋友和家人也因为智能手机通讯而显得互相仇恨。" # Opposite meaning
    ]
    cases["Case3"] = {
        "label": "Case 3 (Smartphone communication)",
        "src_full": src3, "ref_full": ref3,
        "src_segs": src3_segs, "ref_segs": ref3_segs,
        "mt_full_good": "".join(mt3_segs_good), "mt_full_bad": "".join(mt3_segs_bad),
        "mt_segs_good": mt3_segs_good, "mt_segs_bad": mt3_segs_bad
    }

    # Case 4: Iran women (Source 20)
    # Error: "refusing to comply" -> "happily agreeing to comply"
    src4 = "In Iran, for example, women have led the greatest threat to the Islamic government in 40 years simply by refusing to comply with laws that force them to cover their hair."
    ref4 = "以伊朗为例，妇女仅仅因为拒绝遵守强制她们佩戴头巾的法律，就对伊斯兰政府构成了40年来最大的威胁。"
    src4_segs = [
        "In Iran, for example, women have led the greatest threat to the Islamic government in 40 years ",
        "simply by refusing to comply with laws that force them to cover their hair."
    ]
    ref4_segs = [
        "以伊朗为例，妇女就对伊斯兰政府构成了40年来最大的威胁，",
        "仅仅因为拒绝遵守强制她们佩戴头巾的法律。"
    ]
    mt4_segs_good = ref4_segs
    mt4_segs_bad = [
        "以伊朗为例，妇女就对伊斯兰政府构成了40年来最大的威胁，",
        "仅仅因为她们非常开心地同意遵守强制佩戴头巾的法律。" # Contradictory/Sarcastic error
    ]
    cases["Case4"] = {
        "label": "Case 4 (Iran protests)",
        "src_full": src4, "ref_full": ref4,
        "src_segs": src4_segs, "ref_segs": ref4_segs,
        "mt_full_good": "".join(mt4_segs_good), "mt_full_bad": "".join(mt4_segs_bad),
        "mt_segs_good": mt4_segs_good, "mt_segs_bad": mt4_segs_bad
    }

    # Case 5: Bangladesh GDP (Source 25)
    # Error: "more than tripled" -> "collapsed to zero"
    src5 = "Its gross domestic product more than tripled. The change gathered its own momentum."
    ref5 = "孟加拉国国内生产总值增长了两倍多。这种变化积聚了动力。"
    src5_segs = [
        "Its gross domestic product more than tripled. ",
        "The change gathered its own momentum."
    ]
    ref5_segs = [
        "孟加拉国国内生产总值增长了两倍多。",
        "这种变化积聚了动力。"
    ]
    mt5_segs_good = ref5_segs
    mt5_segs_bad = [
        "孟加拉国国内生产总值彻底崩溃归零。", # Numerical/Factual error
        "这种变化积聚了动力。"
    ]
    cases["Case5"] = {
        "label": "Case 5 (Bangladesh GDP)",
        "src_full": src5, "ref_full": ref5,
        "src_segs": src5_segs, "ref_segs": ref5_segs,
        "mt_full_good": "".join(mt5_segs_good), "mt_full_bad": "".join(mt5_segs_bad),
        "mt_segs_good": mt5_segs_good, "mt_segs_bad": mt5_segs_bad
    }

    # Case 6: Rail strike cost (Source 34)
    # Error: "€100m a day" -> "a few pennies a year"
    src6 = "The rail strike could cost businesses €100m ($110m) a day if it forced them to interrupt production."
    ref6 = "如果铁路罢工迫使企业中断生产，每天可能损失1亿欧元（1.1亿美元）。"
    src6_segs = [
        "The rail strike could cost businesses €100m ($110m) a day ",
        "if it forced them to interrupt production."
    ]
    ref6_segs = [
        "铁路罢工每天可能给企业造成1亿欧元（1.1亿美元）的损失，",
        "如果它迫使企业中断生产。"
    ]
    mt6_segs_good = ref6_segs
    mt6_segs_bad = [
        "铁路罢工每年可能只给企业造成几分钱的损失，", # Magnitude error
        "如果它迫使企业中断生产。"
    ]
    cases["Case6"] = {
        "label": "Case 6 (Strike cost)",
        "src_full": src6, "ref_full": ref6,
        "src_segs": src6_segs, "ref_segs": ref6_segs,
        "mt_full_good": "".join(mt6_segs_good), "mt_full_bad": "".join(mt6_segs_bad),
        "mt_segs_good": mt6_segs_good, "mt_segs_bad": mt6_segs_bad
    }

    # Case 7: Beer vs Wine (Source 44)
    # Error: "bears no comparison" -> "is exactly the same"
    src7 = "He told Euronews that a market dominated by multinational brewing conglomerates bears no comparison with one comprised almost entirely of small, domestic producers."
    ref7 = "他告诉欧洲新闻台，由跨国啤酒集团主导的市场与几乎完全由小型国内酿造商组成的市场完全没有可比性。"
    src7_segs = [
        "He told Euronews that a market dominated by multinational brewing conglomerates ",
        "bears no comparison with one comprised almost entirely of small, domestic producers."
    ]
    ref7_segs = [
        "他告诉欧洲新闻台，由跨国啤酒集团主导的市场，",
        "与几乎完全由小型国内酿造商组成的市场完全没有可比性。"
    ]
    mt7_segs_good = ref7_segs
    mt7_segs_bad = [
        "他告诉欧洲新闻台，由跨国啤酒集团主导的市场，",
        "与几乎完全由小型国内酿造商组成的市场是一模一样的。" # Logical opposite
    ]
    cases["Case7"] = {
        "label": "Case 7 (Beer market comparison)",
        "src_full": src7, "ref_full": ref7,
        "src_segs": src7_segs, "ref_segs": ref7_segs,
        "mt_full_good": "".join(mt7_segs_good), "mt_full_bad": "".join(mt7_segs_bad),
        "mt_segs_good": mt7_segs_good, "mt_segs_bad": mt7_segs_bad
    }

    # Case 8: Reuse recycling (Source 48)
    # Error: "support" -> "strongly oppose"
    src8 = "They support reuse recycling systems already in place in many countries, he said."
    ref8 = "他说，他们支持许多国家已经实施的再利用回收系统。"
    src8_segs = [
        "They support reuse recycling systems ",
        "already in place in many countries, he said."
    ]
    ref8_segs = [
        "他们支持再利用回收系统，",
        "他说这些系统已经在许多国家实施。"
    ]
    mt8_segs_good = ref8_segs
    mt8_segs_bad = [
        "他们强烈反对并试图破坏再利用回收系统，", # Action reversal
        "他说这些系统已经在许多国家实施。"
    ]
    cases["Case8"] = {
        "label": "Case 8 (Recycling support)",
        "src_full": src8, "ref_full": ref8,
        "src_segs": src8_segs, "ref_segs": ref8_segs,
        "mt_full_good": "".join(mt8_segs_good), "mt_full_bad": "".join(mt8_segs_bad),
        "mt_segs_good": mt8_segs_good, "mt_segs_bad": mt8_segs_bad
    }

    # Case 9: EPA deferred cases (Source 53)
    # Error: "fewer cases" -> "a billion more cases"
    src9 = "In fiscal year 2022, the EPA deferred fewer cases to DOJ (88) than at any point since 2000."
    ref9 = "2022财年，美国环保局向美国司法部提交的案件数量（88）比2000财年以来的任何年份都要少。"
    src9_segs = [
        "In fiscal year 2022, the EPA deferred fewer cases to DOJ (88) ",
        "than at any point since 2000."
    ]
    ref9_segs = [
        "2022财年，美国环保局向美国司法部提交的案件数量（88）",
        "比2000财年以来的任何年份都要少。"
    ]
    mt9_segs_good = ref9_segs
    mt9_segs_bad = [
        "2022财年，美国环保局向美国司法部提交了十亿起案件，", # Exaggeration/Hallucination
        "比2000财年以来的任何年份都要少。"
    ]
    cases["Case9"] = {
        "label": "Case 9 (EPA cases)",
        "src_full": src9, "ref_full": ref9,
        "src_segs": src9_segs, "ref_segs": ref9_segs,
        "mt_full_good": "".join(mt9_segs_good), "mt_full_bad": "".join(mt9_segs_bad),
        "mt_segs_good": mt9_segs_good, "mt_segs_bad": mt9_segs_bad
    }

    # Case 10: Johnson's stance (Source 58)
    # Error: "would not support" -> "would die to support"
    src10 = "Johnson has publicly stated that he would not support a \"continuing resolution\" (CR) to extend the deadline."
    ref10 = "约翰逊曾公开表示，他不会支持“持续决议案”，将截止日期延长。"
    src10_segs = [
        "Johnson has publicly stated that he would not support a \"continuing resolution\" (CR) ",
        "to extend the deadline."
    ]
    ref10_segs = [
        "约翰逊曾公开表示，他不会支持“持续决议案”，",
        "将截止日期延长。"
    ]
    mt10_segs_good = ref10_segs
    mt10_segs_bad = [
        "约翰逊曾公开表示，他死也要支持“持续决议案”，", # Negation error
        "将截止日期延长。"
    ]
    cases["Case10"] = {
        "label": "Case 10 (Johnson CR support)",
        "src_full": src10, "ref_full": ref10,
        "src_segs": src10_segs, "ref_segs": ref10_segs,
        "mt_full_good": "".join(mt10_segs_good), "mt_full_bad": "".join(mt10_segs_bad),
        "mt_segs_good": mt10_segs_good, "mt_segs_bad": mt10_segs_bad
    }

    # Case 11: Bitcoin endorsement (Source 69)
    # Error: "did not approve or endorse" -> "wholeheartedly endorsed and bought"
    src11 = "\"While we approved the listing... we did not approve or endorse Bitcoin,\" SEC Chair Gary Gensler said in a statement."
    ref11 = "“虽然我们批准了上市……但没有批准或认可比特币。”美国证监会主席加里·詹斯勒在一份声明中表示。"
    src11_segs = [
        "\"While we approved the listing... we did not approve or endorse Bitcoin,\" ",
        "SEC Chair Gary Gensler said in a statement."
    ]
    ref11_segs = [
        "“虽然我们批准了上市……但没有批准或认可比特币。”",
        "美国证监会主席加里·詹斯勒在一份声明中表示。"
    ]
    mt11_segs_good = ref11_segs
    mt11_segs_bad = [
        "“虽然我们批准了上市……我们全心全意认可并购买了比特币。”", # Fabrication
        "美国证监会主席加里·詹斯勒在一份声明中表示。"
    ]
    cases["Case11"] = {
        "label": "Case 11 (Bitcoin endorsement)",
        "src_full": src11, "ref_full": ref11,
        "src_segs": src11_segs, "ref_segs": ref11_segs,
        "mt_full_good": "".join(mt11_segs_good), "mt_full_bad": "".join(mt11_segs_bad),
        "mt_segs_good": mt11_segs_good, "mt_segs_bad": mt11_segs_bad
    }

    # Case 12: Kiev taxi driver (Source 77)
    # Error: "serviced his clients in Russian" -> "drove a spaceship to Mars"
    src12 = "Kremen also confirmed that a 3,400 hryvnia ($89) fine had been imposed on a Kiev taxi driver who serviced his clients in Russian."
    ref12 = "克雷门还证实，基辅一名出租车司机用俄语为客户提供服务，被处以3400格里夫纳（89美元）的罚款。"
    src12_segs = [
        "Kremen also confirmed that a 3,400 hryvnia ($89) fine had been imposed on a Kiev taxi driver ",
        "who serviced his clients in Russian."
    ]
    ref12_segs = [
        "克雷门还证实，基辅一名出租车司机被处以3400格里夫纳（89美元）的罚款，",
        "因为他用俄语为客户提供服务。"
    ]
    mt12_segs_good = ref12_segs
    mt12_segs_bad = [
        "克雷门还证实，基辅一名出租车司机被处以3400格里夫纳（89美元）的罚款，",
        "因为他开着宇宙飞船去了火星。" # Complete hallucination
    ]
    cases["Case12"] = {
        "label": "Case 12 (Kiev taxi driver)",
        "src_full": src12, "ref_full": ref12,
        "src_segs": src12_segs, "ref_segs": ref12_segs,
        "mt_full_good": "".join(mt12_segs_good), "mt_full_bad": "".join(mt12_segs_bad),
        "mt_segs_good": mt12_segs_good, "mt_segs_bad": mt12_segs_bad
    }

    # Case 13: Russian language ban (Source 80)
    # Error: "complete bans" -> "mandatory requirements"
    src13 = "Local authorities have introduced complete bans on Russian-language works of art, performances, books, films, songs."
    ref13 = "地方当局全面禁止俄语艺术作品、表演、书籍、电影、歌曲。"
    src13_segs = [
        "Local authorities have introduced complete bans ",
        "on Russian-language works of art, performances, books, films, songs."
    ]
    ref13_segs = [
        "地方当局实施了全面禁令，",
        "针对俄语艺术作品、表演、书籍、电影、歌曲。"
    ]
    mt13_segs_good = ref13_segs
    mt13_segs_bad = [
        "地方当局实施了强制性推广要求，", # Opposite policy
        "针对俄语艺术作品、表演、书籍、电影、歌曲。"
    ]
    cases["Case13"] = {
        "label": "Case 13 (Russian ban)",
        "src_full": src13, "ref_full": ref13,
        "src_segs": src13_segs, "ref_segs": ref13_segs,
        "mt_full_good": "".join(mt13_segs_good), "mt_full_bad": "".join(mt13_segs_bad),
        "mt_segs_good": mt13_segs_good, "mt_segs_bad": mt13_segs_bad
    }

    # Case 14: NHS wait times (Source 83)
    # Error: "83-hour wait" -> "5-minute wait"
    src14 = "An 83-hour wait in a hospital A&E; four-in-ten patients waiting longer than four hours."
    ref14 = "医院急诊室的候诊时间长达83小时；十分之四的患者候诊时间超过4小时。"
    src14_segs = [
        "An 83-hour wait in a hospital A&E; ",
        "four-in-ten patients waiting longer than four hours."
    ]
    ref14_segs = [
        "医院急诊室的候诊时间长达83小时；",
        "十分之四的患者候诊时间超过4小时。"
    ]
    mt14_segs_good = ref14_segs
    mt14_segs_bad = [
        "医院急诊室的候诊时间仅为5分钟；", # Number/Reality distortion
        "十分之四的患者候诊时间超过4小时。"
    ]
    cases["Case14"] = {
        "label": "Case 14 (NHS wait)",
        "src_full": src14, "ref_full": ref14,
        "src_segs": src14_segs, "ref_segs": ref14_segs,
        "mt_full_good": "".join(mt14_segs_good), "mt_full_bad": "".join(mt14_segs_bad),
        "mt_segs_good": mt14_segs_good, "mt_segs_bad": mt14_segs_bad
    }

    # Case 15: Waste prevented tons (Source 92)
    # Error: "prevented almost 1.35 million tonnes" -> "created 10 tonnes"
    src15 = "Over the past six years, the firm's site near Livingston has prevented almost 1.35 million tonnes of waste going to landfill."
    ref15 = "在过去六年里，该公司位于利文斯顿附近的工地避免了近135万吨废物进入垃圾填埋场。"
    src15_segs = [
        "Over the past six years, the firm's site near Livingston ",
        "has prevented almost 1.35 million tonnes of waste going to landfill."
    ]
    ref15_segs = [
        "在过去六年里，该公司位于利文斯顿附近的工地",
        "避免了近135万吨废物进入垃圾填埋场。"
    ]
    mt15_segs_good = ref15_segs
    mt15_segs_bad = [
        "在过去六年里，该公司位于利文斯顿附近的工地",
        "仅仅产生了10吨废物进入垃圾填埋场。" # Numerical minimization error
    ]
    cases["Case15"] = {
        "label": "Case 15 (Waste tons)",
        "src_full": src15, "ref_full": ref15,
        "src_segs": src15_segs, "ref_segs": ref15_segs,
        "mt_full_good": "".join(mt15_segs_good), "mt_full_bad": "".join(mt15_segs_bad),
        "mt_segs_good": mt15_segs_good, "mt_segs_bad": mt15_segs_bad
    }

    # Case 16: Rent increases (Source 99)
    # Error: "increased in all sized properties" -> "decreased everywhere"
    src16 = "What we find now from the latest Scottish Government statistics is that, in the last year alone, average rents have increased in all sized properties."
    ref16 = "我们从苏格兰政府的最新统计数据中发现，仅去年一年，所有大小房产的平均租金都有所上涨。"
    src16_segs = [
        "What we find now from the latest Scottish Government statistics is that, ",
        "in the last year alone, average rents have increased in all sized properties."
    ]
    ref16_segs = [
        "我们从苏格兰政府的最新统计数据中发现，",
        "仅去年一年，所有大小房产的平均租金都有所上涨。"
    ]
    mt16_segs_good = ref16_segs
    mt16_segs_bad = [
        "我们从苏格兰政府的最新统计数据中发现，",
        "仅去年一年，所有大小房产的平均租金都大幅下降了。" # Opposite trend
    ]
    cases["Case16"] = {
        "label": "Case 16 (Rent stats)",
        "src_full": src16, "ref_full": ref16,
        "src_segs": src16_segs, "ref_segs": ref16_segs,
        "mt_full_good": "".join(mt16_segs_good), "mt_full_bad": "".join(mt16_segs_bad),
        "mt_segs_good": mt16_segs_good, "mt_segs_bad": mt16_segs_bad
    }

    # Case 17: Rent cap warning (Source 101)
    # Error: "warnings... went unheeded" -> "warnings... were obeyed"
    src17 = "There were plenty of warnings from all involved in the housing sector that this would happen, but these went unheeded."
    ref17 = "所有房地产参与者都曾多次发出警告，称这种情况将会发生，但这些警告没有得到重视。"
    src17_segs = [
        "There were plenty of warnings from all involved in the housing sector that this would happen, ",
        "but these went unheeded."
    ]
    ref17_segs = [
        "所有房地产参与者都曾多次发出警告，称这种情况将会发生，",
        "但这些警告没有得到重视。"
    ]
    mt17_segs_good = ref17_segs
    mt17_segs_bad = [
        "所有房地产参与者都曾多次发出警告，称这种情况将会发生，",
        "而且这些警告得到了严格的遵守和执行。" # Fact reversal
    ]
    cases["Case17"] = {
        "label": "Case 17 (Rent warnings)",
        "src_full": src17, "ref_full": ref17,
        "src_segs": src17_segs, "ref_segs": ref17_segs,
        "mt_full_good": "".join(mt17_segs_good), "mt_full_bad": "".join(mt17_segs_bad),
        "mt_segs_good": mt17_segs_good, "mt_segs_bad": mt17_segs_bad
    }

    # Case 18: Greggs prices (Source 113)
    # Error: "no plans currently to up prices" -> "plans to double prices"
    src18 = "Chief executive Roisin Currie told PA she has \"no plans currently\" to up prices at the till as it expects a more stable cost base."
    ref18 = "首席执行官罗伊辛·库里告诉PA，公司预计成本基础将更加稳定，她“目前没有计划”提高价格。"
    src18_segs = [
        "Chief executive Roisin Currie told PA she has \"no plans currently\" to up prices at the till ",
        "as it expects a more stable cost base."
    ]
    ref18_segs = [
        "首席执行官罗伊辛·库里告诉PA，她“目前没有计划”提高价格，",
        "因为公司预计成本基础将更加稳定。"
    ]
    mt18_segs_good = ref18_segs
    mt18_segs_bad = [
        "首席执行官罗伊辛·库里告诉PA，她计划将价格翻倍，", # Semantic opposite
        "因为公司预计成本基础将更加稳定。"
    ]
    cases["Case18"] = {
        "label": "Case 18 (Greggs pricing)",
        "src_full": src18, "ref_full": ref18,
        "src_segs": src18_segs, "ref_segs": ref18_segs,
        "mt_full_good": "".join(mt18_segs_good), "mt_full_bad": "".join(mt18_segs_bad),
        "mt_segs_good": mt18_segs_good, "mt_segs_bad": mt18_segs_bad
    }

    # Case 19: Post Office scandal (Source 123)
    # Error: "honest person treated as a criminal" -> "criminal treated as a hero"
    src19 = "People were sent to prison, lost homes, marriages and reputations, and endured the mental torment of an honest person treated as a criminal."
    ref19 = "这些经理被送进监狱，家庭、婚姻分崩瓦解、名誉受损，并承受着诚实人被当作罪犯对待的精神折磨。"
    src19_segs = [
        "People were sent to prison, lost homes, marriages and reputations, ",
        "and endured the mental torment of an honest person treated as a criminal."
    ]
    ref19_segs = [
        "这些人被送进监狱，家庭、婚姻分崩瓦解、名誉受损，",
        "并承受着诚实人被当作罪犯对待的精神折磨。"
    ]
    mt19_segs_good = ref19_segs
    mt19_segs_bad = [
        "这些人被送进监狱，家庭、婚姻分崩瓦解、名誉受损，",
        "并享受着罪犯被当作英雄对待的精神愉悦。" # Semantic reversal/Irony error
    ]
    cases["Case19"] = {
        "label": "Case 19 (Post Office scandal)",
        "src_full": src19, "ref_full": ref19,
        "src_segs": src19_segs, "ref_segs": ref19_segs,
        "mt_full_good": "".join(mt19_segs_good), "mt_full_bad": "".join(mt19_segs_bad),
        "mt_segs_good": mt19_segs_good, "mt_segs_bad": mt19_segs_bad
    }

    # Case 20: Rishi Sunak's law (Source 128)
    # Error: "dangerous precedent" -> "safe and standard procedure"
    src20 = "A law overturning court verdicts may set a dangerous precedent at a time when populists are testing the boundaries of political power."
    ref20 = "在民粹主义者试探政治权力边界之际，旨在推翻法院判决的法律可能会开创一个危险的先例。"
    src20_segs = [
        "A law overturning court verdicts may set a dangerous precedent ",
        "at a time when populists are testing the boundaries of political power."
    ]
    ref20_segs = [
        "旨在推翻法院判决的法律可能会开创一个危险的先例，",
        "特别是在民粹主义者试探政治权力边界之际。"
    ]
    mt20_segs_good = ref20_segs
    mt20_segs_bad = [
        "旨在推翻法院判决的法律是一个安全且标准的程序，", # Semantic opposite
        "特别是在民粹主义者试探政治权力边界之际。"
    ]
    cases["Case20"] = {
        "label": "Case 20 (Sunak law)",
        "src_full": src20, "ref_full": ref20,
        "src_segs": src20_segs, "ref_segs": ref20_segs,
        "mt_full_good": "".join(mt20_segs_good), "mt_full_bad": "".join(mt20_segs_bad),
        "mt_segs_good": mt20_segs_good, "mt_segs_bad": mt20_segs_bad
    }

    # Case 21: Boeing safety (Source 150)
    # Error: "fewest total hull losses" -> "most total hull losses"
    src21 = "According to the report, which was published in September 2023, the Boeing 787 and Airbus A350 are the aircraft models with the fewest total hull losses."
    ref21 = "根据这份于2023年9月发布的报告，波音787和空客A350是发生机身全损事故最少的机型。"
    src21_segs = [
        "According to the report, which was published in September 2023, ",
        "the Boeing 787 and Airbus A350 are the aircraft models with the fewest total hull losses."
    ]
    ref21_segs = [
        "根据这份于2023年9月发布的报告，",
        "波音787和空客A350是发生机身全损事故最少的机型。"
    ]
    mt21_segs_good = ref21_segs
    mt21_segs_bad = [
        "根据这份于2023年9月发布的报告，",
        "波音787和空客A350是发生机身全损事故最多的机型。" # Semantic Opposite
    ]
    cases["Case21"] = {
        "label": "Case 21 (Boeing stats)",
        "src_full": src21, "ref_full": ref21,
        "src_segs": src21_segs, "ref_segs": ref21_segs,
        "mt_full_good": "".join(mt21_segs_good), "mt_full_bad": "".join(mt21_segs_bad),
        "mt_segs_good": mt21_segs_good, "mt_segs_bad": mt21_segs_bad
    }

    # Case 22: Lazy Job (Source 175)
    # Error: "lazy" -> "hardworking"
    src22 = "I guess I should find a job, but I'm quite lazy. Hire me now, I'll drink all your beer."
    ref22 = "我想我应该找份工作，但是我很懒。现在就雇我吧，让我帮你喝啤酒。"
    src22_segs = [
        "I guess I should find a job, but I'm quite lazy. ",
        "Hire me now, I'll drink all your beer."
    ]
    ref22_segs = [
        "我想我应该找份工作，但是我很懒。",
        "现在就雇我吧，让我帮你喝啤酒。"
    ]
    mt22_segs_good = ref22_segs
    mt22_segs_bad = [
        "我想我应该找份工作，但是我非常勤奋。", # Semantic Opposite
        "现在就雇我吧，让我帮你喝啤酒。"
    ]
    cases["Case22"] = {
        "label": "Case 22 (Lazy job)",
        "src_full": src22, "ref_full": ref22,
        "src_segs": src22_segs, "ref_segs": ref22_segs,
        "mt_full_good": "".join(mt22_segs_good), "mt_full_bad": "".join(mt22_segs_bad),
        "mt_segs_good": mt22_segs_good, "mt_segs_bad": mt22_segs_bad
    }

    # Case 23: Surgeon call (Source 193)
    # Error: "snow was forecast" -> "it was a sunny day"
    src23 = "He called the day before because snow was forecast, and as the medical team was fussing over me as I was coming out of anesthesia."
    ref23 = "因为天气预报有雪，所以他在前一天就打了电话。当我在恢复室从麻醉中苏醒时，医疗团队正手忙脚乱给我做检查。"
    src23_segs = [
        "He called the day before because snow was forecast, ",
        "and as the medical team was fussing over me as I was coming out of anesthesia."
    ]
    ref23_segs = [
        "因为天气预报有雪，所以他在前一天就打了电话。",
        "当我在恢复室从麻醉中苏醒时，医疗团队正手忙脚乱给我做检查。"
    ]
    mt23_segs_good = ref23_segs
    mt23_segs_bad = [
        "因为天气预报是大晴天，所以他在前一天就打了电话。", # Hallucination/Error
        "当我在恢复室从麻醉中苏醒时，医疗团队正手忙脚乱给我做检查。"
    ]
    cases["Case23"] = {
        "label": "Case 23 (Surgeon call)",
        "src_full": src23, "ref_full": ref23,
        "src_segs": src23_segs, "ref_segs": ref23_segs,
        "mt_full_good": "".join(mt23_segs_good), "mt_full_bad": "".join(mt23_segs_bad),
        "mt_segs_good": mt23_segs_good, "mt_segs_bad": mt23_segs_bad
    }

    # Case 24: Ancestry DNA (Source 225)
    # Error: "British and Irish" -> "Japanese and Korean"
    src24 = "The results are in and it's completely consistent with my family history. 89.9% British and Irish with most of that concentrated in the scottish borders."
    ref24 = "结果出来了，与家族史完全一致。89.9%的英国人和爱尔兰人，其中大部分集中在苏格兰边境。"
    src24_segs = [
        "The results are in and it's completely consistent with my family history. ",
        "89.9% British and Irish with most of that concentrated in the scottish borders."
    ]
    ref24_segs = [
        "结果出来了，与家族史完全一致。",
        "89.9%的英国人和爱尔兰人，其中大部分集中在苏格兰边境。"
    ]
    mt24_segs_good = ref24_segs
    mt24_segs_bad = [
        "结果出来了，与家族史完全一致。",
        "89.9%的日本人和韩国人，其中大部分集中在苏格兰边境。" # Factual Error/Hallucination
    ]
    cases["Case24"] = {
        "label": "Case 24 (Ancestry DNA)",
        "src_full": src24, "ref_full": ref24,
        "src_segs": src24_segs, "ref_segs": ref24_segs,
        "mt_full_good": "".join(mt24_segs_good), "mt_full_bad": "".join(mt24_segs_bad),
        "mt_segs_good": mt24_segs_good, "mt_segs_bad": mt24_segs_bad
    }

    # Case 25: Tyvek suit (Source 248)
    # Error: "super good call" -> "terrible idea"
    src25 = "The tyvek suit… super good call. Despite getting really hot and sweaty and overall nasty."
    ref25 = "特卫强套装……好主意。尽管热得汗流浃背，浑身上下都是污垢。"
    src25_segs = [
        "The tyvek suit… super good call. ",
        "Despite getting really hot and sweaty and overall nasty."
    ]
    ref25_segs = [
        "特卫强套装……好主意。",
        "尽管热得汗流浃背，浑身上下都是污垢。"
    ]
    mt25_segs_good = ref25_segs
    mt25_segs_bad = [
        "特卫强套装……真是个馊主意。", # Semantic Opposite
        "尽管热得汗流浃背，浑身上下都是污垢。"
    ]
    cases["Case25"] = {
        "label": "Case 25 (Tyvek suit)",
        "src_full": src25, "ref_full": ref25,
        "src_segs": src25_segs, "ref_segs": ref25_segs,
        "mt_full_good": "".join(mt25_segs_good), "mt_full_bad": "".join(mt25_segs_bad),
        "mt_segs_good": mt25_segs_good, "mt_segs_bad": mt25_segs_bad
    }

    # Case 26: Battery runtime (Source 281)
    # Error: "4 hours" -> "4 years"
    src26 = "On a full charge, it gives me a runtime of about 4 hours. I call this a feature."
    ref26 = "充满电后，它可以运行四小时。我称之为“功能”。"
    src26_segs = [
        "On a full charge, it gives me a runtime of about 4 hours. ",
        "I call this a feature."
    ]
    ref26_segs = [
        "充满电后，它可以运行四小时。",
        "我称之为“功能”。"
    ]
    mt26_segs_good = ref26_segs
    mt26_segs_bad = [
        "充满电后，它可以运行四年。", # Numerical/Logical Error
        "我称之为“功能”。"
    ]
    cases["Case26"] = {
        "label": "Case 26 (Battery runtime)",
        "src_full": src26, "ref_full": ref26,
        "src_segs": src26_segs, "ref_segs": ref26_segs,
        "mt_full_good": "".join(mt26_segs_good), "mt_full_bad": "".join(mt26_segs_bad),
        "mt_segs_good": mt26_segs_good, "mt_segs_bad": mt26_segs_bad
    }

    # Case 27: Football meltdown (Source 288)
    # Error: "officiating hamstrung them" -> "officiating helped them win"
    src27 = "Absolute meltdown. Obviously, the officiating hamstrung them, but the last twenty minutes are the play of the team that did not deserve to win."
    ref27 = "绝对的崩盘。很明显，裁判限制了球员的发挥，但最后20分钟的表现表明，球队本不应该获胜。"
    src27_segs = [
        "Absolute meltdown. ",
        "Obviously, the officiating hamstrung them, but the last twenty minutes are the play of the team that did not deserve to win."
    ]
    ref27_segs = [
        "绝对的崩盘。",
        "很明显，裁判限制了球员的发挥，但最后20分钟的表现表明，球队本不应该获胜。"
    ]
    mt27_segs_good = ref27_segs
    mt27_segs_bad = [
        "绝对的崩盘。",
        "很明显，裁判帮助他们赢得了比赛，但最后20分钟的表现表明，球队本不应该获胜。" # Semantic Error
    ]
    cases["Case27"] = {
        "label": "Case 27 (Football meltdown)",
        "src_full": src27, "ref_full": ref27,
        "src_segs": src27_segs, "ref_segs": ref27_segs,
        "mt_full_good": "".join(mt27_segs_good), "mt_full_bad": "".join(mt27_segs_bad),
        "mt_segs_good": mt27_segs_good, "mt_segs_bad": mt27_segs_bad
    }

    # Case 28: New website (Source 311)
    # Error: "new website" -> "new car"
    src28 = "I will need a new website and am making a list of the accessibility and privacy requirements I want."
    ref28 = "我需要建立一个新网站，目前正在列出我想要的可访问性和隐私要求。"
    src28_segs = [
        "I will need a new website ",
        "and am making a list of the accessibility and privacy requirements I want."
    ]
    ref28_segs = [
        "我需要建立一个新网站，",
        "目前正在列出我想要的可访问性和隐私要求。"
    ]
    mt28_segs_good = ref28_segs
    mt28_segs_bad = [
        "我需要买一辆新车，", # Entity Error
        "目前正在列出我想要的可访问性和隐私要求。"
    ]
    cases["Case28"] = {
        "label": "Case 28 (New website)",
        "src_full": src28, "ref_full": ref28,
        "src_segs": src28_segs, "ref_segs": ref28_segs,
        "mt_full_good": "".join(mt28_segs_good), "mt_full_bad": "".join(mt28_segs_bad),
        "mt_segs_good": mt28_segs_good, "mt_segs_bad": mt28_segs_bad
    }

    # Case 29: FAA medical (Source 341)
    # Error: "difficult time" -> "easy time"
    src29 = "I'm likely going to have a VERY difficult time getting a medical clearance due to the FAA's stance on certain medications."
    ref29 = "由于美国联邦航空局对某些药物的立场，我可能很难获得健康证明。"
    src29_segs = [
        "I'm likely going to have a VERY difficult time getting a medical clearance ",
        "due to the FAA's stance on certain medications."
    ]
    ref29_segs = [
        "我可能很难获得健康证明，",
        "由于美国联邦航空局对某些药物的立场。"
    ]
    mt29_segs_good = ref29_segs
    mt29_segs_bad = [
        "我可能很容易就能获得健康证明，", # Semantic Opposite
        "由于美国联邦航空局对某些药物的立场。"
    ]
    cases["Case29"] = {
        "label": "Case 29 (FAA medical)",
        "src_full": src29, "ref_full": ref29,
        "src_segs": src29_segs, "ref_segs": ref29_segs,
        "mt_full_good": "".join(mt29_segs_good), "mt_full_bad": "".join(mt29_segs_bad),
        "mt_segs_good": mt29_segs_good, "mt_segs_bad": mt29_segs_bad
    }

    # Case 30: Rowing goal (Source 366)
    # Error: "real goal" -> "fake news"
    src30 = "You know it's a real goal when there's a spreadsheet to track it. Another month down and I'm still ahead of schedule."
    ref30 = "用电子表格来跟踪目标，你就知道目标是真实的。又一个月过去了，目前进度仍然超前。"
    src30_segs = [
        "You know it's a real goal when there's a spreadsheet to track it. ",
        "Another month down and I'm still ahead of schedule."
    ]
    ref30_segs = [
        "用电子表格来跟踪目标，你就知道目标是真实的。",
        "又一个月过去了，目前进度仍然超前。"
    ]
    mt30_segs_good = ref30_segs
    mt30_segs_bad = [
        "用电子表格来跟踪目标，你就知道这只是假新闻。", # Semantic distortion
        "又一个月过去了，目前进度仍然超前。"
    ]
    cases["Case30"] = {
        "label": "Case 30 (Rowing goal)",
        "src_full": src30, "ref_full": ref30,
        "src_segs": src30_segs, "ref_segs": ref30_segs,
        "mt_full_good": "".join(mt30_segs_good), "mt_full_bad": "".join(mt30_segs_bad),
        "mt_segs_good": mt30_segs_good, "mt_segs_bad": mt30_segs_bad
    }

    # Case 31: HTML streaming (Source 395)
    # Error: "very long web page" -> "short tweet"
    src31 = "Imagine a very long web page with lots of sections (e.g., single-page HTML spec). If the user requests a specific section..."
    ref31 = "想象一下，网页很长，分很多部分（如单页HTML规范）。如果用户请求特定部分..."
    src31_segs = [
        "Imagine a very long web page with lots of sections (e.g., single-page HTML spec). ",
        "If the user requests a specific section..."
    ]
    ref31_segs = [
        "想象一下，网页很长，分很多部分（如单页HTML规范）。",
        "如果用户请求特定部分..."
    ]
    mt31_segs_good = ref31_segs
    mt31_segs_bad = [
        "想象一下，一条非常短的推文，几乎没有任何内容。", # Semantic Opposite/Hallucination
        "如果用户请求特定部分..."
    ]
    cases["Case31"] = {
        "label": "Case 31 (HTML streaming)",
        "src_full": src31, "ref_full": ref31,
        "src_segs": src31_segs, "ref_segs": ref31_segs,
        "mt_full_good": "".join(mt31_segs_good), "mt_full_bad": "".join(mt31_segs_bad),
        "mt_segs_good": mt31_segs_good, "mt_segs_bad": mt31_segs_bad
    }

    # Case 32: Venice (Source 426)
    # Error: "attracts millions" -> "scares away everyone"
    src32 = "A place of beauty and romance that attracts millions of visitors every year. Just a few steps from the well-trodden tourist routes..."
    ref32 = "一个美丽而又浪漫的地方，每年吸引了数百万游客。距离这个热门景点仅几步之遥..."
    src32_segs = [
        "A place of beauty and romance that attracts millions of visitors every year. ",
        "Just a few steps from the well-trodden tourist routes..."
    ]
    ref32_segs = [
        "一个美丽而又浪漫的地方，每年吸引了数百万游客。",
        "距离这个热门景点仅几步之遥..."
    ]
    mt32_segs_good = ref32_segs
    mt32_segs_bad = [
        "一个丑陋而恐怖的地方，每年把所有游客都吓跑了。", # Semantic Opposite
        "距离这个热门景点仅几步之遥..."
    ]
    cases["Case32"] = {
        "label": "Case 32 (Venice)",
        "src_full": src32, "ref_full": ref32,
        "src_segs": src32_segs, "ref_segs": ref32_segs,
        "mt_full_good": "".join(mt32_segs_good), "mt_full_bad": "".join(mt32_segs_bad),
        "mt_segs_good": mt32_segs_good, "mt_segs_bad": mt32_segs_bad
    }

    # Case 33: Japanese Giant (Source 442)
    # Error: "authentic and real" -> "fake and computer-generated"
    src33 = "This is an authentic and real old footage, of a true certain giant individual from Japan, walking in some military parade."
    ref33 = "这是一个真实的老镜头，一位真正的日本巨人，正走在某次阅兵式上。"
    src33_segs = [
        "This is an authentic and real old footage, of a true certain giant individual from Japan, ",
        "walking in some military parade."
    ]
    ref33_segs = [
        "这是一个真实的老镜头，一位真正的日本巨人，",
        "正走在某次阅兵式上。"
    ]
    mt33_segs_good = ref33_segs
    mt33_segs_bad = [
        "这是一个虚假且由电脑合成的新镜头，一位虚构的日本巨人，", # Semantic Opposite
        "正走在某次阅兵式上。"
    ]
    cases["Case33"] = {
        "label": "Case 33 (Japanese giant)",
        "src_full": src33, "ref_full": ref33,
        "src_segs": src33_segs, "ref_segs": ref33_segs,
        "mt_full_good": "".join(mt33_segs_good), "mt_full_bad": "".join(mt33_segs_bad),
        "mt_segs_good": mt33_segs_good, "mt_segs_bad": mt33_segs_bad
    }

    # Case 34: Skin Potion (Source 472)
    # Error: "moisturize and illuminate" -> "burn and darken"
    src34 = "Skin potion is handcrafted with organic and all natural ingredients that deeply moisturize and illuminate the skin."
    ref34 = "Skinpotion使用全天然有机成分手工打造而成，能深层滋润和提亮肌肤。"
    src34_segs = [
        "Skin potion is handcrafted with organic and all natural ingredients ",
        "that deeply moisturize and illuminate the skin."
    ]
    ref34_segs = [
        "Skinpotion使用全天然有机成分手工打造而成，",
        "能深层滋润和提亮肌肤。"
    ]
    mt34_segs_good = ref34_segs
    mt34_segs_bad = [
        "Skinpotion使用全天然有机成分手工打造而成，",
        "会严重烧伤并使肌肤变黑。" # Semantic Opposite/Harmful
    ]
    cases["Case34"] = {
        "label": "Case 34 (Skin Potion)",
        "src_full": src34, "ref_full": ref34,
        "src_segs": src34_segs, "ref_segs": ref34_segs,
        "mt_full_good": "".join(mt34_segs_good), "mt_full_bad": "".join(mt34_segs_bad),
        "mt_segs_good": mt34_segs_good, "mt_segs_bad": mt34_segs_bad
    }

    # Case 35: Media trust (Source 490)
    # Error: "some value" -> "no value"
    src35 = "Now that being said, I still think there's some value to the mainstream media. Just because they are exaggerated and biased..."
    ref35 = "即便是这样，我仍然认为主流媒体还是有一些价值的。他们夸大其词、存有偏见..."
    src35_segs = [
        "Now that being said, I still think there's some value to the mainstream media. ",
        "Just because they are exaggerated and biased..."
    ]
    ref35_segs = [
        "即便是这样，我仍然认为主流媒体还是有一些价值的。",
        "他们夸大其词、存有偏见..."
    ]
    mt35_segs_good = ref35_segs
    mt35_segs_bad = [
        "即便是这样，我仍然认为主流媒体毫无价值。", # Semantic Opposite
        "他们夸大其词、存有偏见..."
    ]
    cases["Case35"] = {
        "label": "Case 35 (Media value)",
        "src_full": src35, "ref_full": ref35,
        "src_segs": src35_segs, "ref_segs": ref35_segs,
        "mt_full_good": "".join(mt35_segs_good), "mt_full_bad": "".join(mt35_segs_bad),
        "mt_segs_good": mt35_segs_good, "mt_segs_bad": mt35_segs_bad
    }

    # Case 36: Grout repair (Source 511)
    # Error: "14, 15 years old" -> "brand new"
    src36 = "Hello, it's Bayano with Bayano Reno. We're going to prepare some grout. The tars looks to be about 14, 15 years old."
    ref36 = "大家好，我是Bahiano装修公司的Bahiano。接下来我们要修补泥浆。这些瓷砖看起来有十四五年了。"
    src36_segs = [
        "Hello, it's Bayano with Bayano Reno. We're going to prepare some grout. ",
        "The tars looks to be about 14, 15 years old."
    ]
    ref36_segs = [
        "大家好，我是Bahiano装修公司的Bahiano。接下来我们要修补泥浆。",
        "这些瓷砖看起来有十四五年了。"
    ]
    mt36_segs_good = ref36_segs
    mt36_segs_bad = [
        "大家好，我是Bahiano装修公司的Bahiano。接下来我们要修补泥浆。",
        "这些瓷砖看起来是全新的。" # Semantic Opposite
    ]
    cases["Case36"] = {
        "label": "Case 36 (Grout age)",
        "src_full": src36, "ref_full": ref36,
        "src_segs": src36_segs, "ref_segs": ref36_segs,
        "mt_full_good": "".join(mt36_segs_good), "mt_full_bad": "".join(mt36_segs_bad),
        "mt_segs_good": mt36_segs_good, "mt_segs_bad": mt36_segs_bad
    }

    # Case 37: US-China Relations (Source 541)
    # Error: "without being enemies" -> "while being enemies"
    src37 = "We will have differences in the future. But what we must do is to find a way to see that we can have differences without being enemies in war."
    ref37 = "将来也会有分歧。而我们要做的，就是设法看到，我们可以有分歧，但不必成为战争中敌对的双方。"
    src37_segs = [
        "We will have differences in the future. ",
        "But what we must do is to find a way to see that we can have differences without being enemies in war."
    ]
    ref37_segs = [
        "将来也会有分歧。",
        "而我们要做的，就是设法看到，我们可以有分歧，但不必成为战争中敌对的双方。"
    ]
    mt37_segs_good = ref37_segs
    mt37_segs_bad = [
        "将来也会有分歧。",
        "而我们要做的，就是设法看到，我们必须在战争中成为死敌。" # Semantic Opposite
    ]
    cases["Case37"] = {
        "label": "Case 37 (US-China)",
        "src_full": src37, "ref_full": ref37,
        "src_segs": src37_segs, "ref_segs": ref37_segs,
        "mt_full_good": "".join(mt37_segs_good), "mt_full_bad": "".join(mt37_segs_bad),
        "mt_segs_good": mt37_segs_good, "mt_segs_bad": mt37_segs_bad
    }

    # Case 38: Chemical Alert (Source 572)
    # Error: "alert was lifted" -> "alert was intensified"
    src38 = "Locals were told not to leave their homes and to close all windows and doors, although this alert was lifted around 7 AM..."
    ref38 = "当地居民被告知应待在家里并关闭所有门窗，不过早上7点左右...警报也就随之解除了。"
    src38_segs = [
        "Locals were told not to leave their homes and to close all windows and doors, ",
        "although this alert was lifted around 7 AM..."
    ]
    ref38_segs = [
        "当地居民被告知应待在家里并关闭所有门窗，",
        "不过早上7点左右，警报也就随之解除了。"
    ]
    mt38_segs_good = ref38_segs
    mt38_segs_bad = [
        "当地居民被告知应待在家里并关闭所有门窗，",
        "不过早上7点左右，警报级别被提升到了最高级。" # Semantic Opposite
    ]
    cases["Case38"] = {
        "label": "Case 38 (Chemical alert)",
        "src_full": src38, "ref_full": ref38,
        "src_segs": src38_segs, "ref_segs": ref38_segs,
        "mt_full_good": "".join(mt38_segs_good), "mt_full_bad": "".join(mt38_segs_bad),
        "mt_segs_good": mt38_segs_good, "mt_segs_bad": mt38_segs_bad
    }

    # Case 39: Surgery Age (Source 814)
    # Error: "recent surgery" -> "upcoming wedding"
    src39 = "You've even tied my recent surgery to my age. Well, I got to be honest with you."
    ref39 = "你们甚至把我近期的手术和我的年龄联系起来了。好吧，我得跟你们说实话。"
    src39_segs = [
        "You've even tied my recent surgery to my age. ",
        "Well, I got to be honest with you."
    ]
    ref39_segs = [
        "你们甚至把我近期的手术和我的年龄联系起来了。",
        "好吧，我得跟你们说实话。"
    ]
    mt39_segs_good = ref39_segs
    mt39_segs_bad = [
        "你们甚至把我即将举行的婚礼和我的年龄联系起来了。", # Hallucination
        "好吧，我得跟你们说实话。"
    ]
    cases["Case39"] = {
        "label": "Case 39 (Surgery age)",
        "src_full": src39, "ref_full": ref39,
        "src_segs": src39_segs, "ref_segs": ref39_segs,
        "mt_full_good": "".join(mt39_segs_good), "mt_full_bad": "".join(mt39_segs_bad),
        "mt_segs_good": mt39_segs_good, "mt_segs_bad": mt39_segs_bad
    }

    # Case 40: Travel Work (Source 652)
    # Error: "worked internationally" -> "never worked anywhere"
    src40 = "I've worked internationally. I've worked this continent. I've worked at different states, different market segments."
    ref40 = "我在国际上工作过。在这个大陆工作过。在不同的州、不同的细分市场都工作过。"
    src40_segs = [
        "I've worked internationally. I've worked this continent. ",
        "I've worked at different states, different market segments."
    ]
    ref40_segs = [
        "我在国际上工作过。在这个大陆工作过。",
        "在不同的州、不同的细分市场都工作过。"
    ]
    mt40_segs_good = ref40_segs
    mt40_segs_bad = [
        "我从未在任何地方工作过。我没出过国。", # Semantic Opposite
        "在不同的州、不同的细分市场都工作过。"
    ]
    cases["Case40"] = {
        "label": "Case 40 (Work experience)",
        "src_full": src40, "ref_full": ref40,
        "src_segs": src40_segs, "ref_segs": ref40_segs,
        "mt_full_good": "".join(mt40_segs_good), "mt_full_bad": "".join(mt40_segs_bad),
        "mt_segs_good": mt40_segs_good, "mt_segs_bad": mt40_segs_bad
    }

    # Case 41: Aluminum properties (Source 581)
    # Error: "youngest members" -> "oldest members"
    src41 = "What is it that has made aluminum, one of the youngest members in the family of metals, such an outstanding material?"
    ref41 = "铝是“金属家族”中最年轻的成员之一，是什么让它成为如此出色的材料呢？"
    src41_segs = [
        "What is it that has made aluminum, one of the youngest members in the family of metals, ",
        "such an outstanding material?"
    ]
    ref41_segs = [
        "铝是“金属家族”中最年轻的成员之一，",
        "是什么让它成为如此出色的材料呢？"
    ]
    mt41_segs_good = ref41_segs
    mt41_segs_bad = [
        "铝是“金属家族”中最古老的成员之一，", # Semantic Opposite
        "是什么让它成为如此出色的材料呢？"
    ]
    cases["Case41"] = {
        "label": "Case 41 (Aluminum age)",
        "src_full": src41, "ref_full": ref41,
        "src_segs": src41_segs, "ref_segs": ref41_segs,
        "mt_full_good": "".join(mt41_segs_good), "mt_full_bad": "".join(mt41_segs_bad),
        "mt_segs_good": mt41_segs_good, "mt_segs_bad": mt41_segs_bad
    }

    # Case 42: Milgram application (Source 584)
    # Error: "were rejected" -> "were immediately accepted"
    src42 = "Initial applications to Harvard for a psychology masters were rejected, but was eventually admitted."
    ref42 = "他最初申请哈佛大学心理学硕士学位被拒，但最终还是被录取了。"
    src42_segs = [
        "Initial applications to Harvard for a psychology masters were rejected, ",
        "but was eventually admitted."
    ]
    ref42_segs = [
        "他最初申请哈佛大学心理学硕士学位被拒，",
        "但最终还是被录取了。"
    ]
    mt42_segs_good = ref42_segs
    mt42_segs_bad = [
        "他最初申请哈佛大学心理学硕士学位被立即录取，", # Fact Reversal
        "但最终还是被录取了。"
    ]
    cases["Case42"] = {
        "label": "Case 42 (Milgram Harvard)",
        "src_full": src42, "ref_full": ref42,
        "src_segs": src42_segs, "ref_segs": ref42_segs,
        "mt_full_good": "".join(mt42_segs_good), "mt_full_bad": "".join(mt42_segs_bad),
        "mt_segs_good": mt42_segs_good, "mt_segs_bad": mt42_segs_bad
    }

    # Case 43: Teenager tragedy (Source 597)
    # Error: "is dead" -> "is partying"
    src43 = "A teenager is dead after police found her body in an eastside ditch."
    ref43 = "警方在东区的沟渠里发现了一名少女的尸体。"
    src43_segs = [
        "A teenager is dead ",
        "after police found her body in an eastside ditch."
    ]
    ref43_segs = [
        "一名少女已确认死亡，",
        "警方在东区的沟渠里发现了她的尸体。"
    ]
    mt43_segs_good = ref43_segs
    mt43_segs_bad = [
        "一名少女正在开派对，", # Hallucination/Tone error
        "警方在东区的沟渠里发现了她的尸体。"
    ]
    cases["Case43"] = {
        "label": "Case 43 (Teenager tragedy)",
        "src_full": src43, "ref_full": ref43,
        "src_segs": src43_segs, "ref_segs": ref43_segs,
        "mt_full_good": "".join(mt43_segs_good), "mt_full_bad": "".join(mt43_segs_bad),
        "mt_segs_good": mt43_segs_good, "mt_segs_bad": mt43_segs_bad
    }

    # Case 44: Game Ping (Source 604)
    # Error: "ping is fine" -> "internet is broken"
    src44 = "My ping is fine. Anyways yeah you can see what I mean by the frame rate."
    ref44 = "我的ping没问题。不管怎样。对，你明白我说的帧率是什么意思吧。"
    src44_segs = [
        "My ping is fine. ",
        "Anyways yeah you can see what I mean by the frame rate."
    ]
    ref44_segs = [
        "我的ping没问题。",
        "不管怎样。对，你明白我说的帧率是什么意思吧。"
    ]
    mt44_segs_good = ref44_segs
    mt44_segs_bad = [
        "我的网络完全断开了。", # Semantic Opposite
        "不管怎样。对，你明白我说的帧率是什么意思吧。"
    ]
    cases["Case44"] = {
        "label": "Case 44 (Game ping)",
        "src_full": src44, "ref_full": ref44,
        "src_segs": src44_segs, "ref_segs": ref44_segs,
        "mt_full_good": "".join(mt44_segs_good), "mt_full_bad": "".join(mt44_segs_bad),
        "mt_segs_good": mt44_segs_good, "mt_segs_bad": mt44_segs_bad
    }

    # Case 45: Electronics IC (Source 819)
    # Error: "is dislocated" -> "is perfectly attached"
    src45 = "Now we can see this IC is dislocated from the PCB. We can see the condition of this IC."
    ref45 = "现在我们可以看到，这个IC从PCB板上脱落了。我们可以看到这个IC的状况。"
    src45_segs = [
        "Now we can see this IC is dislocated from the PCB. ",
        "We can see the condition of this IC."
    ]
    ref45_segs = [
        "现在我们可以看到，这个IC从PCB板上脱落了。",
        "我们可以看到这个IC的状况。"
    ]
    mt45_segs_good = ref45_segs
    mt45_segs_bad = [
        "现在我们可以看到，这个IC完美地焊接在PCB板上。", # Semantic Opposite
        "我们可以看到这个IC的状况。"
    ]
    cases["Case45"] = {
        "label": "Case 45 (Electronics IC)",
        "src_full": src45, "ref_full": ref45,
        "src_segs": src45_segs, "ref_segs": ref45_segs,
        "mt_full_good": "".join(mt45_segs_good), "mt_full_bad": "".join(mt45_segs_bad),
        "mt_segs_good": mt45_segs_good, "mt_segs_bad": mt45_segs_bad
    }

    # Case 46: Lalabella journey (Source 630)
    # Error: "1,600 miles" -> "100 meters"
    src46 = "The king, Labella, is said to have travelled the 1,600 miles to Jerusalem."
    ref46 = "据说国王拉利贝拉曾跋涉1600英里来到耶路撒冷。"
    src46_segs = [
        "The king, Labella, is said to have travelled ",
        "the 1,600 miles to Jerusalem."
    ]
    ref46_segs = [
        "据说国王拉利贝拉曾跋涉",
        "1600英里来到耶路撒冷。"
    ]
    mt46_segs_good = ref46_segs
    mt46_segs_bad = [
        "据说国王拉利贝拉曾跋涉",
        "100米来到耶路撒冷。" # Numerical Error
    ]
    cases["Case46"] = {
        "label": "Case 46 (Lalabella distance)",
        "src_full": src46, "ref_full": ref46,
        "src_segs": src46_segs, "ref_segs": ref46_segs,
        "mt_full_good": "".join(mt46_segs_good), "mt_full_bad": "".join(mt46_segs_bad),
        "mt_segs_good": mt46_segs_good, "mt_segs_bad": mt46_segs_bad
    }

    # Case 47: Mars volcano (Source 634)
    # Error: "81,000 feet tall" -> "small hill"
    src47 = "To do that, there's a volcano on Mars, 81,000 feet tall, almost three times taller than Everest."
    ref47 = "火星上有一座火山，高81000英尺——几乎是珠穆朗玛峰的三倍。"
    src47_segs = [
        "To do that, there's a volcano on Mars, 81,000 feet tall, ",
        "almost three times taller than Everest."
    ]
    ref47_segs = [
        "火星上有一座火山，高81000英尺，",
        "几乎是珠穆朗玛峰的三倍。"
    ]
    mt47_segs_good = ref47_segs
    mt47_segs_bad = [
        "火星上有一座小土丘，只有几英尺高，", # Magnitude Error
        "几乎是珠穆朗玛峰的三倍。"
    ]
    cases["Case47"] = {
        "label": "Case 47 (Mars volcano)",
        "src_full": src47, "ref_full": ref47,
        "src_segs": src47_segs, "ref_segs": ref47_segs,
        "mt_full_good": "".join(mt47_segs_good), "mt_full_bad": "".join(mt47_segs_bad),
        "mt_segs_good": mt47_segs_good, "mt_segs_bad": mt47_segs_bad
    }

    # Case 48: Kerensky government (Source 646)
    # Error: "overthrown by" -> "appointed by"
    src48 = "Mr. Alexander Kerensky, president of the Democratic Provisional Government of Russia, overthrown by the Communists in 1917."
    ref48 = "亚历山大·克伦斯基先生，他是1917年被共产党推翻的俄国民主临时政府的总统。"
    src48_segs = [
        "Mr. Alexander Kerensky, president of the Democratic Provisional Government of Russia, ",
        "overthrown by the Communists in 1917."
    ]
    ref48_segs = [
        "亚历山大·克伦斯基先生，他是俄国民主临时政府的总统，",
        "在1917年被共产党推翻。"
    ]
    mt48_segs_good = ref48_segs
    mt48_segs_bad = [
        "亚历山大·克伦斯基先生，他是俄国民主临时政府的总统，",
        "在1917年被共产党任命。" # Historical Error
    ]
    cases["Case48"] = {
        "label": "Case 48 (Kerensky)",
        "src_full": src48, "ref_full": ref48,
        "src_segs": src48_segs, "ref_segs": ref48_segs,
        "mt_full_good": "".join(mt48_segs_good), "mt_full_bad": "".join(mt48_segs_bad),
        "mt_segs_good": mt48_segs_good, "mt_segs_bad": mt48_segs_bad
    }

    # Case 49: Robot hand (Source 660)
    # Error: "not freaked out" -> "terrified"
    src49 = "I'm not freaked out by it's... All right, fine. I'm freaked out."
    ref49 = "我没有被吓到，只是……好的，行。我是被吓坏了。"
    src49_segs = [
        "I'm not freaked out by it's... ",
        "All right, fine. I'm freaked out."
    ]
    ref49_segs = [
        "我没有被吓到，只是……",
        "好的，行。我是被吓坏了。"
    ]
    mt49_segs_good = ref49_segs
    mt49_segs_bad = [
        "我被它吓得半死，天啊……", # Premature admission/Error
        "好的，行。我是被吓坏了。"
    ]
    cases["Case49"] = {
        "label": "Case 49 (Robot hand)",
        "src_full": src49, "ref_full": ref49,
        "src_segs": src49_segs, "ref_segs": ref49_segs,
        "mt_full_good": "".join(mt49_segs_good), "mt_full_bad": "".join(mt49_segs_bad),
        "mt_segs_good": mt49_segs_good, "mt_segs_bad": mt49_segs_bad
    }

    # Case 50: Spine repair (Source 668)
    # Error: "remove this old spine" -> "keep this old spine"
    src50 = "So what I'm going to do first for the spine repair is I need to remove this old spine."
    ref50 = "那么，我首先要做的就是去除旧的脊柱，以便进行修复。"
    src50_segs = [
        "So what I'm going to do first for the spine repair is ",
        "I need to remove this old spine."
    ]
    ref50_segs = [
        "为了修复书脊，我首先要做的就是，",
        "我需要去除这个旧书脊。"
    ]
    mt50_segs_good = ref50_segs
    mt50_segs_bad = [
        "为了修复书脊，我首先要做的就是，",
        "我需要永久保留这个旧书脊。" # Instruction Reversal
    ]
    cases["Case50"] = {
        "label": "Case 50 (Book spine)",
        "src_full": src50, "ref_full": ref50,
        "src_segs": src50_segs, "ref_segs": ref50_segs,
        "mt_full_good": "".join(mt50_segs_good), "mt_full_bad": "".join(mt50_segs_bad),
        "mt_segs_good": mt50_segs_good, "mt_segs_bad": mt50_segs_bad
    }

    # Case 51: Milky Way collision (Source 678)
    # Error: "collided with" -> "avoided"
    src51 = "More than 4.5 billion years ago, the Milky Way galaxy collided with a nearby dwarf galaxy."
    ref51 = "45亿多年前，银河系与附近的矮星系相撞。"
    src51_segs = [
        "More than 4.5 billion years ago, ",
        "the Milky Way galaxy collided with a nearby dwarf galaxy."
    ]
    ref51_segs = [
        "45亿多年前，",
        "银河系与附近的矮星系相撞。"
    ]
    mt51_segs_good = ref51_segs
    mt51_segs_bad = [
        "45亿多年前，",
        "银河系巧妙地避开了附近的矮星系。" # Event Negation
    ]
    cases["Case51"] = {
        "label": "Case 51 (Milky Way collision)",
        "src_full": src51, "ref_full": ref51,
        "src_segs": src51_segs, "ref_segs": ref51_segs,
        "mt_full_good": "".join(mt51_segs_good), "mt_full_bad": "".join(mt51_segs_bad),
        "mt_segs_good": mt51_segs_good, "mt_segs_bad": mt51_segs_bad
    }

    # Case 52: Plant friendship (Source 684)
    # Error: "forming a special friendship" -> "declaring total war"
    src52 = "They do this by forming a special friendship with soil dwelling fungi called mycorrhiza."
    ref52 = "它们通过与一种叫做菌根的土栖真菌建立特殊的盟友关系来做到这一点。"
    src52_segs = [
        "They do this by forming a special friendship ",
        "with soil dwelling fungi called mycorrhiza."
    ]
    ref52_segs = [
        "它们通过建立特殊的盟友关系来做到这一点，",
        "对象是一种叫做菌根的土栖真菌。"
    ]
    mt52_segs_good = ref52_segs
    mt52_segs_bad = [
        "它们通过宣战并试图消灭对方来做到这一点，", # Semantic Opposite
        "对象是一种叫做菌根的土栖真菌。"
    ]
    cases["Case52"] = {
        "label": "Case 52 (Plant fungi)",
        "src_full": src52, "ref_full": ref52,
        "src_segs": src52_segs, "ref_segs": ref52_segs,
        "mt_full_good": "".join(mt52_segs_good), "mt_full_bad": "".join(mt52_segs_bad),
        "mt_segs_good": mt52_segs_good, "mt_segs_bad": mt52_segs_bad
    }

    # Case 53: Rotterdam architecture (Source 693)
    # Error: "modern and urban-looking" -> "ancient and rural"
    src53 = "It is a city that divides opinions because of its more modern and urban- looking architecture."
    ref53 = "因为更加现代、更具都市风情的建筑而令人们对它褒贬不一。"
    src53_segs = [
        "It is a city that divides opinions ",
        "because of its more modern and urban- looking architecture."
    ]
    ref53_segs = [
        "这座城市令人们褒贬不一，",
        "因为其更加现代、更具都市风情的建筑。"
    ]
    mt53_segs_good = ref53_segs
    mt53_segs_bad = [
        "这座城市令人们褒贬不一，",
        "因为其古老且充满乡村气息的建筑。" # Semantic Opposite
    ]
    cases["Case53"] = {
        "label": "Case 53 (Rotterdam)",
        "src_full": src53, "ref_full": ref53,
        "src_segs": src53_segs, "ref_segs": ref53_segs,
        "mt_full_good": "".join(mt53_segs_good), "mt_full_bad": "".join(mt53_segs_bad),
        "mt_segs_good": mt53_segs_good, "mt_segs_bad": mt53_segs_bad
    }

    # Case 54: Roofing advice (Source 706)
    # Error: "never shingled before" -> "shingled for 50 years"
    src54 = "The subject of this video is, if I was to do this roof, and I've never shingled before, how would I do it?"
    ref54 = "视频的主题是，如果我要做这个屋顶，而我以前从来没有贴过瓦板，我该怎么办？"
    src54_segs = [
        "The subject of this video is, if I was to do this roof, ",
        "and I've never shingled before, how would I do it?"
    ]
    ref54_segs = [
        "视频的主题是，如果我要做这个屋顶，",
        "而我以前从来没有贴过瓦板，我该怎么办？"
    ]
    mt54_segs_good = ref54_segs
    mt54_segs_bad = [
        "视频的主题是，如果我要做这个屋顶，",
        "而我已经贴了50年瓦板，是世界级专家，我该怎么办？" # Contradiction/Hallucination
    ]
    cases["Case54"] = {
        "label": "Case 54 (Roofing intro)",
        "src_full": src54, "ref_full": ref54,
        "src_segs": src54_segs, "ref_segs": ref54_segs,
        "mt_full_good": "".join(mt54_segs_good), "mt_full_bad": "".join(mt54_segs_bad),
        "mt_segs_good": mt54_segs_good, "mt_segs_bad": mt54_segs_bad
    }

    # Case 55: Unknown Soldier Tomb (Source 715)
    # Error: "guarded 24 hours a day" -> "left unguarded"
    src55 = "The Tomb of the Unknown Soldier has been guarded 24 hours a day, seven days a week, 365 days a year, for the past 77 years."
    ref55 = "过去77年里，无名战士墓每年365天、每周7天、每周24小时都有人把守。"
    src55_segs = [
        "The Tomb of the Unknown Soldier has been guarded 24 hours a day, ",
        "seven days a week, 365 days a year, for the past 77 years."
    ]
    ref55_segs = [
        "无名战士墓全天24小时有人把守，",
        "过去77年里，每年365天、每周7天都是如此。"
    ]
    mt55_segs_good = ref55_segs
    mt55_segs_bad = [
        "无名战士墓全天24小时无人看管，", # Semantic Opposite
        "过去77年里，每年365天、每周7天都是如此。"
    ]
    cases["Case55"] = {
        "label": "Case 55 (Tomb Guard)",
        "src_full": src55, "ref_full": ref55,
        "src_segs": src55_segs, "ref_segs": ref55_segs,
        "mt_full_good": "".join(mt55_segs_good), "mt_full_bad": "".join(mt55_segs_bad),
        "mt_segs_good": mt55_segs_good, "mt_segs_bad": mt55_segs_bad
    }

    # Case 56: Yima Underground City (Source 722)
    # Error: "protect his people" -> "destroy his people"
    src56 = "The Zoroastrian text, Veda, states that Yema built an underground city on the orders of the god Arua Mazda, to protect his people from a catastrophic winner."
    ref56 = "琐罗亚斯德教经典《文迪达》中说，Yima遵照Ahura Mazda神的命令建造了一座地下城市，以保护他的人民度过一个极其艰难的冬天。"
    src56_segs = [
        "The Zoroastrian text, Veda, states that Yema built an underground city on the orders of the god Arua Mazda, ",
        "to protect his people from a catastrophic winner."
    ]
    ref56_segs = [
        "琐罗亚斯德教经典《文迪达》中说，Yima遵照Ahura Mazda神的命令建造了一座地下城市，",
        "以保护他的人民度过一个极其艰难的冬天。"
    ]
    mt56_segs_good = ref56_segs
    mt56_segs_bad = [
        "琐罗亚斯德教经典《文迪达》中说，Yima遵照Ahura Mazda神的命令建造了一座地下城市，",
        "以彻底消灭他的人民并引发灾难。" # Semantic Opposite
    ]
    cases["Case56"] = {
        "label": "Case 56 (Yima city)",
        "src_full": src56, "ref_full": ref56,
        "src_segs": src56_segs, "ref_segs": ref56_segs,
        "mt_full_good": "".join(mt56_segs_good), "mt_full_bad": "".join(mt56_segs_bad),
        "mt_segs_good": mt56_segs_good, "mt_segs_bad": mt56_segs_bad
    }

    # Case 57: ROG Phone 5 (Source 726)
    # Error: "taking back the crown" -> "losing miserably"
    src57 = "In this video, we're going to find out if the ROG Phone 5 is truly taking back the crown as the king of gaming."
    ref57 = "在这个视频中，我们要去看看ROG Phone 5是否真的夺回了“游戏手机之王”的宝座。"
    src57_segs = [
        "In this video, we're going to find out ",
        "if the ROG Phone 5 is truly taking back the crown as the king of gaming."
    ]
    ref57_segs = [
        "在这个视频中，我们要去看看",
        "ROG Phone 5是否真的夺回了“游戏手机之王”的宝座。"
    ]
    mt57_segs_good = ref57_segs
    mt57_segs_bad = [
        "在这个视频中，我们要去看看",
        "ROG Phone 5是否真的惨败并成为了垃圾手机。" # Negative Sentiment injection
    ]
    cases["Case57"] = {
        "label": "Case 57 (ROG Phone)",
        "src_full": src57, "ref_full": ref57,
        "src_segs": src57_segs, "ref_segs": ref57_segs,
        "mt_full_good": "".join(mt57_segs_good), "mt_full_bad": "".join(mt57_segs_bad),
        "mt_segs_good": mt57_segs_good, "mt_segs_bad": mt57_segs_bad
    }

    # Case 58: Ultralight Cooking (Source 735)
    # Error: "boiling water" -> "freezing water"
    src58 = "We're primarily boiling water so that we can rehydrate our food."
    ref58 = "我们主要是烧水来给食物补充水分。"
    src58_segs = [
        "We're primarily boiling water ",
        "so that we can rehydrate our food."
    ]
    ref58_segs = [
        "我们主要是烧开水，",
        "以便给食物补充水分。"
    ]
    mt58_segs_good = ref58_segs
    mt58_segs_bad = [
        "我们主要是把水冷冻成冰，", # Semantic Opposite
        "以便给食物补充水分。"
    ]
    cases["Case58"] = {
        "label": "Case 58 (Cooking water)",
        "src_full": src58, "ref_full": ref58,
        "src_segs": src58_segs, "ref_segs": ref58_segs,
        "mt_full_good": "".join(mt58_segs_good), "mt_full_bad": "".join(mt58_segs_bad),
        "mt_segs_good": mt58_segs_good, "mt_segs_bad": mt58_segs_bad
    }

    # Case 59: Snowball Earth (Source 739)
    # Error: "prevented sunlight" -> "increased sunlight"
    src59 = "The thick cloud cover prevented sunlight from reaching the surface of the Earth."
    ref59 = "厚厚的云层阻挡了阳光到达地球表面。"
    src59_segs = [
        "The thick cloud cover ",
        "prevented sunlight from reaching the surface of the Earth."
    ]
    ref59_segs = [
        "厚厚的云层",
        "阻挡了阳光到达地球表面。"
    ]
    mt59_segs_good = ref59_segs
    mt59_segs_bad = [
        "厚厚的云层",
        "让阳光更加强烈地照射到地球表面。" # Semantic Opposite
    ]
    cases["Case59"] = {
        "label": "Case 59 (Snowball Earth)",
        "src_full": src59, "ref_full": ref59,
        "src_segs": src59_segs, "ref_segs": ref59_segs,
        "mt_full_good": "".join(mt59_segs_good), "mt_full_bad": "".join(mt59_segs_bad),
        "mt_segs_good": mt59_segs_good, "mt_segs_bad": mt59_segs_bad
    }

    # Case 60: Fake News Study (Source 746)
    # Error: "only 25%" -> "every single one"
    src60 = "According to a Stanford study, only 25 % of high school students were able to identify an accurate news story compared to a fake one."
    ref60 = "根据斯坦福大学的一项研究，只有25%的高中生能够识别准确的新闻报道与假新闻。"
    src60_segs = [
        "According to a Stanford study, only 25 % of high school students ",
        "were able to identify an accurate news story compared to a fake one."
    ]
    ref60_segs = [
        "根据斯坦福大学的一项研究，只有25%的高中生",
        "能够识别准确的新闻报道与假新闻。"
    ]
    mt60_segs_good = ref60_segs
    mt60_segs_bad = [
        "根据斯坦福大学的一项研究，每一个高中生", # Numerical/Factual Error
        "能够识别准确的新闻报道与假新闻。"
    ]
    cases["Case60"] = {
        "label": "Case 60 (Fake news stats)",
        "src_full": src60, "ref_full": ref60,
        "src_segs": src60_segs, "ref_segs": ref60_segs,
        "mt_full_good": "".join(mt60_segs_good), "mt_full_bad": "".join(mt60_segs_bad),
        "mt_segs_good": mt60_segs_good, "mt_segs_bad": mt60_segs_bad
    }


    """
    Constructed cases from WMT24 en-zh data for XCOMET strategy comparison
    Selected 20 NEW examples focusing on LONGER sentences with MULTIPLE segments (3+).
    No duplicates from previous batches.
    """

    # Case 61: Government Knowledge (Source 761)
    # Error: "doesn't work for us" -> "works perfectly for us"
    src61 = "I did some research to get ready for tonight, because the only thing I know about government is the reason we're here, which is that it doesn't work for us."
    ref61 = "我为今晚做了一些研究，因为我对政府唯一的了解就是我们聚在这里的原因，那就是政府不为我们服务。"
    src61_segs = [
        "I did some research to get ready for tonight, ",
        "because the only thing I know about government is the reason we're here, ",
        "which is that it doesn't work for us."
    ]
    ref61_segs = [
        "我为今晚做了一些研究，",
        "因为我对政府唯一的了解就是我们聚在这里的原因，",
        "那就是政府不为我们服务。"
    ]
    mt61_segs_good = ref61_segs
    mt61_segs_bad = [
        "我为今晚做了一些研究，",
        "因为我对政府唯一的了解就是我们聚在这里的原因，",
        "那就是政府完美地为我们服务。" # Semantic Reversal
    ]
    cases["Case61"] = {
        "label": "Case 61 (Government View)",
        "src_full": src61, "ref_full": ref61,
        "src_segs": src61_segs, "ref_segs": ref61_segs,
        "mt_full_good": "".join(mt61_segs_good), "mt_full_bad": "".join(mt61_segs_bad),
        "mt_segs_good": mt61_segs_good, "mt_segs_bad": mt61_segs_bad
    }

    # Case 62: Resident Evil (Source 764)
    # Error: "can't get any worse" -> "was the best masterpiece"
    src62 = "I remember a few years back, watching some of those Resident Evil movies with Mila Jovovich and thinking, I can't get any worse than this."
    ref62 = "我记得几年前，观看米拉·乔沃维奇主演的几部《生化危机》系列电影时想：“没有比这更糟糕的了！”"
    src62_segs = [
        "I remember a few years back, ",
        "watching some of those Resident Evil movies with Mila Jovovich and thinking, ",
        "I can't get any worse than this."
    ]
    ref62_segs = [
        "我记得几年前，",
        "观看米拉·乔沃维奇主演的几部《生化危机》系列电影时想：",
        "“没有比这更糟糕的了！”"
    ]
    mt62_segs_good = ref62_segs
    mt62_segs_bad = [
        "我记得几年前，",
        "观看米拉·乔沃维奇主演的几部《生化危机》系列电影时想：",
        "“这简直是无与伦比的电影杰作！”" # Sentiment/Fact Reversal
    ]
    cases["Case62"] = {
        "label": "Case 62 (Resident Evil)",
        "src_full": src62, "ref_full": ref62,
        "src_segs": src62_segs, "ref_segs": ref62_segs,
        "mt_full_good": "".join(mt62_segs_good), "mt_full_bad": "".join(mt62_segs_bad),
        "mt_segs_good": mt62_segs_good, "mt_segs_bad": mt62_segs_bad
    }

    # Case 63: Raptor Attitude (Source 773)
    # Error: "arrogant and ignorant" -> "humble and enlightened"
    src63 = "For Raptor specialist Bob Anderson, the pioneer's attitude was simply arrogant and ignorant, as all birds of prey were just considered vermin."
    ref63 = "对于猛禽专家鲍勃•安德森来说，这位拓荒者的态度简直是既傲慢又无知，因为所有猛禽都被认为是有害的。"
    src63_segs = [
        "For Raptor specialist Bob Anderson, ",
        "the pioneer's attitude was simply arrogant and ignorant, ",
        "as all birds of prey were just considered vermin."
    ]
    ref63_segs = [
        "对于猛禽专家鲍勃•安德森来说，",
        "这位拓荒者的态度简直是既傲慢又无知，",
        "因为所有猛禽都被认为是有害的。"
    ]
    mt63_segs_good = ref63_segs
    mt63_segs_bad = [
        "对于猛禽专家鲍勃•安德森来说，",
        "这位拓荒者的态度简直是既谦逊又开明，", # Opposite Personality
        "因为所有猛禽都被认为是有害的。"
    ]
    cases["Case63"] = {
        "label": "Case 63 (Raptor Attitude)",
        "src_full": src63, "ref_full": ref63,
        "src_segs": src63_segs, "ref_segs": ref63_segs,
        "mt_full_good": "".join(mt63_segs_good), "mt_full_bad": "".join(mt63_segs_bad),
        "mt_segs_good": mt63_segs_good, "mt_segs_bad": mt63_segs_bad
    }

    # Case 64: Antique Ornament (Source 781)
    # Error: "looked really antique" -> "looked like plastic"
    src64 = "I just continued that process until I had as many layers as I liked, and it looked really antique in like an actual metal ornament."
    ref64 = "我重复同样的步骤，直到弄出想要的层数，它看起来真的很有历史感，就像一件真正的金属装饰品。"
    src64_segs = [
        "I just continued that process until I had as many layers as I liked, ",
        "and it looked really antique ",
        "in like an actual metal ornament."
    ]
    ref64_segs = [
        "我重复同样的步骤，直到弄出想要的层数，",
        "它看起来真的很有历史感，",
        "就像一件真正的金属装饰品。"
    ]
    mt64_segs_good = ref64_segs
    mt64_segs_bad = [
        "我重复同样的步骤，直到弄出想要的层数，",
        "它看起来完全像是廉价的塑料玩具，", # Description Error
        "就像一件真正的金属装饰品。"
    ]
    cases["Case64"] = {
        "label": "Case 64 (Antique Ornament)",
        "src_full": src64, "ref_full": ref64,
        "src_segs": src64_segs, "ref_segs": ref64_segs,
        "mt_full_good": "".join(mt64_segs_good), "mt_full_bad": "".join(mt64_segs_bad),
        "mt_segs_good": mt64_segs_good, "mt_segs_bad": mt64_segs_bad
    }

    # Case 65: Humanity Advancement (Source 785)
    # Error: "never ceased" -> "completely stopped"
    src65 = "The advancement of Humanity never ceased, even for a moment–during difficult times we grow and adapt once again."
    ref65 = "人类从未停止前进的脚步，即便是片刻——我们在困境中成长、调整，然后涅槃重生。"
    src65_segs = [
        "The advancement of Humanity never ceased, ",
        "even for a moment–",
        "during difficult times we grow and adapt once again."
    ]
    ref65_segs = [
        "人类从未停止前进的脚步，",
        "即便是片刻——",
        "我们在困境中成长、调整，然后涅槃重生。"
    ]
    mt65_segs_good = ref65_segs
    mt65_segs_bad = [
        "人类前进的脚步完全停止了，", # Meaning Reversal
        "即便是片刻——",
        "我们在困境中成长、调整，然后涅槃重生。"
    ]
    cases["Case65"] = {
        "label": "Case 65 (Humanity Advancement)",
        "src_full": src65, "ref_full": ref65,
        "src_segs": src65_segs, "ref_segs": ref65_segs,
        "mt_full_good": "".join(mt65_segs_good), "mt_full_bad": "".join(mt65_segs_bad),
        "mt_segs_good": mt65_segs_good, "mt_segs_bad": mt65_segs_bad
    }

    # Case 66: Exploring Paltricus (Source 800)
    # Error: "privilege to those" -> "death sentence for those"
    src66 = "The cold, rainy night ensued my long overdue walk–exploring the city of Paltricus was a privilege to those who’d been safe from the mutation."
    ref66 = "直到寒冷的雨夜降临，我走了很久——探索Paltricus这座城市对于那些免于突变的幸存者而言真是一种殊荣。"
    src66_segs = [
        "The cold, rainy night ensued my long overdue walk–",
        "exploring the city of Paltricus was a privilege ",
        "to those who’d been safe from the mutation."
    ]
    ref66_segs = [
        "直到寒冷的雨夜降临，我走了很久——",
        "探索Paltricus这座城市真是一种殊荣，",
        "对于那些免于突变的幸存者而言。"
    ]
    mt66_segs_good = ref66_segs
    mt66_segs_bad = [
        "直到寒冷的雨夜降临，我走了很久——",
        "探索Paltricus这座城市简直是死刑判决，", # Semantic Opposite
        "对于那些免于突变的幸存者而言。"
    ]
    cases["Case66"] = {
        "label": "Case 66 (Paltricus Walk)",
        "src_full": src66, "ref_full": ref66,
        "src_segs": src66_segs, "ref_segs": ref66_segs,
        "mt_full_good": "".join(mt66_segs_good), "mt_full_bad": "".join(mt66_segs_bad),
        "mt_segs_good": mt66_segs_good, "mt_segs_bad": mt66_segs_bad
    }

    # Case 67: Dangerous Neighborhood (Source 820)
    # Error: "unless they wanted to get mugged" -> "because it was the safest place"
    src67 = "Quick regret flushed over me as I heard footsteps close behind, because nobody hung around this place unless they wanted to get mugged or beaten up."
    ref67 = "听到身后的脚步声时，我立刻后悔了，因为除非想被抢劫或被殴打，否则没有人会在这种地方闲逛。"
    src67_segs = [
        "Quick regret flushed over me as I heard footsteps close behind, ",
        "because nobody hung around this place ",
        "unless they wanted to get mugged or beaten up."
    ]
    ref67_segs = [
        "听到身后的脚步声时，我立刻后悔了，",
        "因为没有人会在这种地方闲逛，",
        "除非想被抢劫或被殴打。"
    ]
    mt67_segs_good = ref67_segs
    mt67_segs_bad = [
        "听到身后的脚步声时，我立刻后悔了，",
        "因为每个人都喜欢在这里闲逛，",
        "毕竟这是全城最安全的地方。" # Logic Reversal
    ]
    cases["Case67"] = {
        "label": "Case 67 (Dangerous Street)",
        "src_full": src67, "ref_full": ref67,
        "src_segs": src67_segs, "ref_segs": ref67_segs,
        "mt_full_good": "".join(mt67_segs_good), "mt_full_bad": "".join(mt67_segs_bad),
        "mt_segs_good": mt67_segs_good, "mt_segs_bad": mt67_segs_bad
    }

    # Case 68: Alborn Trouble (Source 830)
    # Error: "gotten into trouble" -> "become best friends"
    src68 = "No need to stick my nose out for someone, especially when they seem like they’ve just gotten into trouble with every authority in the city."
    ref68 = "我没有必要为别人出头，何况这些家伙看起来好像刚刚找过本市所有当局的麻烦。"
    src68_segs = [
        "No need to stick my nose out for someone, ",
        "especially when they seem like ",
        "they’ve just gotten into trouble with every authority in the city."
    ]
    ref68_segs = [
        "我没有必要为别人出头，",
        "何况这些家伙看起来好像，",
        "刚刚找过本市所有当局的麻烦。"
    ]
    mt68_segs_good = ref68_segs
    mt68_segs_bad = [
        "我没有必要为别人出头，",
        "何况这些家伙看起来好像，",
        "刚刚成为了本市所有当局的好朋友。" # Semantic Opposite
    ]
    cases["Case68"] = {
        "label": "Case 68 (Alborn Trouble)",
        "src_full": src68, "ref_full": ref68,
        "src_segs": src68_segs, "ref_segs": ref68_segs,
        "mt_full_good": "".join(mt68_segs_good), "mt_full_bad": "".join(mt68_segs_bad),
        "mt_segs_good": mt68_segs_good, "mt_segs_bad": mt68_segs_bad
    }

    # Case 69: Alcatraz Name (Source 840)
    # Error: "fit the name pretty well" -> "was a terrible name choice"
    src69 = "I’d done numerous kinds of research on the outer cities, and the capital of the infected, Alcatraz, felt like it fit the name pretty well."
    ref69 = "我曾对外围城区和被感染区首都Alcatraz做过很多调查，感觉这个名字还挺贴切。"
    src69_segs = [
        "I’d done numerous kinds of research on the outer cities, ",
        "and the capital of the infected, Alcatraz, ",
        "felt like it fit the name pretty well."
    ]
    ref69_segs = [
        "我曾对外围城区做过很多调查，",
        "还有被感染区首都Alcatraz，",
        "感觉这个名字还挺贴切。"
    ]
    mt69_segs_good = ref69_segs
    mt69_segs_bad = [
        "我曾对外围城区做过很多调查，",
        "还有被感染区首都Alcatraz，",
        "感觉这个名字起得真是糟糕透顶。" # Opinion Reversal
    ]
    cases["Case69"] = {
        "label": "Case 69 (Alcatraz Name)",
        "src_full": src69, "ref_full": ref69,
        "src_segs": src69_segs, "ref_segs": ref69_segs,
        "mt_full_good": "".join(mt69_segs_good), "mt_full_bad": "".join(mt69_segs_bad),
        "mt_segs_good": mt69_segs_good, "mt_segs_bad": mt69_segs_bad
    }

    # Case 70: Thassalin Fire (Source 868)
    # Error: "set fire to" -> "watered"
    src70 = "A burst of flames erupted from Thassalin's jaws as the colossal Thraki set fire to a very specific patch of forest, then circled above."
    ref70 = "萨萨林从嘴中喷出火焰，高大的斯拉克人点燃了一片森林，然后在上空盘旋。"
    src70_segs = [
        "A burst of flames erupted from Thassalin's jaws ",
        "as the colossal Thraki set fire to a very specific patch of forest, ",
        "then circled above."
    ]
    ref70_segs = [
        "萨萨林从嘴中喷出火焰，",
        "高大的斯拉克人点燃了一片森林，",
        "然后在上空盘旋。"
    ]
    mt70_segs_good = ref70_segs
    mt70_segs_bad = [
        "萨萨林从嘴中喷出火焰，",
        "高大的斯拉克人给一片森林浇了浇水，", # Action Substitution/Hallucination
        "然后在上空盘旋。"
    ]
    cases["Case70"] = {
        "label": "Case 70 (Thassalin Fire)",
        "src_full": src70, "ref_full": ref70,
        "src_segs": src70_segs, "ref_segs": ref70_segs,
        "mt_full_good": "".join(mt70_segs_good), "mt_full_bad": "".join(mt70_segs_bad),
        "mt_segs_good": mt70_segs_good, "mt_segs_bad": mt70_segs_bad
    }

    # Case 71: Kayel's Claws (Source 870)
    # Error: "dug into... bleed" -> "gently massaged"
    src71 = "Kayel clearly didn't agree with Nyssi, he looked less black than normal and his claws had dug into Thassalin's back to the point that he'd made the Thraki bleed."
    ref71 = "卡耶尔显然不同意尼西的说法，他看起来不像正常人那么黑，他的爪子已经深深地扎进了萨萨林的背上，弄得鲜血直流。"
    src71_segs = [
        "Kayel clearly didn't agree with Nyssi, ",
        "he looked less black than normal ",
        "and his claws had dug into Thassalin's back to the point that he'd made the Thraki bleed."
    ]
    ref71_segs = [
        "卡耶尔显然不同意尼西的说法，",
        "他看起来不像正常人那么黑，",
        "他的爪子已经深深地扎进了萨萨林的背上，弄得鲜血直流。"
    ]
    mt71_segs_good = ref71_segs
    mt71_segs_bad = [
        "卡耶尔显然不同意尼西的说法，",
        "他看起来不像正常人那么黑，",
        "他正用爪子温柔地给萨萨林按摩背部。" # Action Reversal/Tone Error
    ]
    cases["Case71"] = {
        "label": "Case 71 (Kayel Claws)",
        "src_full": src71, "ref_full": ref71,
        "src_segs": src71_segs, "ref_segs": ref71_segs,
        "mt_full_good": "".join(mt71_segs_good), "mt_full_bad": "".join(mt71_segs_bad),
        "mt_segs_good": mt71_segs_good, "mt_segs_bad": mt71_segs_bad
    }

    # Case 72: Darkness Monster (Source 898)
    # Error: "went dark" -> "became bright"
    src72 = "Snarling and hissing, the creature flew upwards, spitting into the air, and everything suddenly went dark, as if the monster was sucking the light from the sky."
    ref72 = "它咆哮着，喉咙嘶嘶作响，径直向上飞去，边向空中喷水，突然四周变得漆黑一片，仿佛怪物正在攫取天空中的光明。"
    src72_segs = [
        "Snarling and hissing, the creature flew upwards, spitting into the air, ",
        "and everything suddenly went dark, ",
        "as if the monster was sucking the light from the sky."
    ]
    ref72_segs = [
        "它咆哮着，喉咙嘶嘶作响，径直向上飞去，边向空中喷水，",
        "突然四周变得漆黑一片，",
        "仿佛怪物正在攫取天空中的光明。"
    ]
    mt72_segs_good = ref72_segs
    mt72_segs_bad = [
        "它咆哮着，喉咙嘶嘶作响，径直向上飞去，边向空中喷水，",
        "突然四周变得光芒万丈，", # Semantic Opposite
        "仿佛怪物正在攫取天空中的光明。"
    ]
    cases["Case72"] = {
        "label": "Case 72 (Darkness Monster)",
        "src_full": src72, "ref_full": ref72,
        "src_segs": src72_segs, "ref_segs": ref72_segs,
        "mt_full_good": "".join(mt72_segs_good), "mt_full_bad": "".join(mt72_segs_bad),
        "mt_segs_good": mt72_segs_good, "mt_segs_bad": mt72_segs_bad
    }

    # Case 73: Healing Powers (Source 901)
    # Error: "healing powers kicked in" -> "powers failed completely"
    src73 = "She was pretty sure she had broken something, but her weird, unnatural healing powers had already kicked in."
    ref73 = "她想自己一定摔坏了什么地方，但那奇怪、非自然的治愈能力已经开始发挥作用。"
    src73_segs = [
        "She was pretty sure ",
        "she had broken something, ",
        "but her weird, unnatural healing powers had already kicked in."
    ]
    ref73_segs = [
        "她想自己一定，",
        "摔坏了什么地方，",
        "但那奇怪、非自然的治愈能力已经开始发挥作用。"
    ]
    mt73_segs_good = ref73_segs
    mt73_segs_bad = [
        "她想自己一定，",
        "摔坏了什么地方，",
        "遗憾的是她那奇怪的治愈能力完全失效了。" # Fact Reversal
    ]
    cases["Case73"] = {
        "label": "Case 73 (Healing Powers)",
        "src_full": src73, "ref_full": ref73,
        "src_segs": src73_segs, "ref_segs": ref73_segs,
        "mt_full_good": "".join(mt73_segs_good), "mt_full_bad": "".join(mt73_segs_bad),
        "mt_segs_good": mt73_segs_good, "mt_segs_bad": mt73_segs_bad
    }

    # Case 74: Surprise Leap (Source 916)
    # Error: "something screeched" -> "a puppy barked"
    src74 = "Before Tenuk and Kayel could ask what the fuck they were looking at, something screeched and leaped out from a nearby bush."
    ref74 = "还没等特努克和卡耶尔想明白看到了什么，一个东西尖叫着，从附近的灌木丛里蹿了出来。"
    src74_segs = [
        "Before Tenuk and Kayel could ask ",
        "what the fuck they were looking at, ",
        "something screeched and leaped out from a nearby bush."
    ]
    ref74_segs = [
        "还没等特努克和卡耶尔开口问，",
        "他们究竟看到了什么鬼东西，",
        "一个东西尖叫着，从附近的灌木丛里蹿了出来。"
    ]
    mt74_segs_good = ref74_segs
    mt74_segs_bad = [
        "还没等特努克和卡耶尔开口问，",
        "他们究竟看到了什么鬼东西，",
        "一只可爱的小狗汪汪叫着从灌木丛里走了出来。" # Hallucination/Tone Error
    ]
    cases["Case74"] = {
        "label": "Case 74 (Surprise Leap)",
        "src_full": src74, "ref_full": ref74,
        "src_segs": src74_segs, "ref_segs": ref74_segs,
        "mt_full_good": "".join(mt74_segs_good), "mt_full_bad": "".join(mt74_segs_bad),
        "mt_segs_good": mt74_segs_good, "mt_segs_bad": mt74_segs_bad
    }

    # Case 75: Eirwen Execution (Source 949)
    # Error: "most feared person" -> "most beloved clown"
    src75 = "As Eirwen approached the wooden pole, the most feared person in the kingdom showed up: The fire user."
    ref75 = "当艾尔文走近木柱时，王国里最可怕的人出现了：他就是火刑的行刑者。"
    src75_segs = [
        "As Eirwen approached the wooden pole, ",
        "the most feared person in the kingdom showed up: ",
        "The fire user."
    ]
    ref75_segs = [
        "当艾尔文走近木柱时，",
        "王国里最可怕的人出现了：",
        "他就是火刑的行刑者。"
    ]
    mt75_segs_good = ref75_segs
    mt75_segs_bad = [
        "当艾尔文走近木柱时，",
        "王国里最受人喜爱的小丑出现了：", # Hallucination/Character Error
        "他就是火刑的行刑者。"
    ]
    cases["Case75"] = {
        "label": "Case 75 (Execution)",
        "src_full": src75, "ref_full": ref75,
        "src_segs": src75_segs, "ref_segs": ref75_segs,
        "mt_full_good": "".join(mt75_segs_good), "mt_full_bad": "".join(mt75_segs_bad),
        "mt_segs_good": mt75_segs_good, "mt_segs_bad": mt75_segs_bad
    }

    # Case 76: Room Description (Source 965)
    # Error: "desk was broken" -> "desk was pristine"
    src76 = "Papers were everywhere, books were scattered about, the desk was broken, and there were multiple ice sculptures scattered about."
    ref76 = "房间里到处都是纸片，书本散落一地，书桌破败不堪，还有很多冰雕。"
    src76_segs = [
        "Papers were everywhere, ",
        "books were scattered about, the desk was broken, ",
        "and there were multiple ice sculptures scattered about."
    ]
    ref76_segs = [
        "房间里到处都是纸片，",
        "书本散落一地，书桌破败不堪，",
        "还有很多冰雕散落在各处。"
    ]
    mt76_segs_good = ref76_segs
    mt76_segs_bad = [
        "房间里到处都是纸片，",
        "书本整整齐齐，书桌也是崭新的一尘不染，", # Description Reversal
        "还有很多冰雕散落在各处。"
    ]
    cases["Case76"] = {
        "label": "Case 76 (Messy Room)",
        "src_full": src76, "ref_full": ref76,
        "src_segs": src76_segs, "ref_segs": ref76_segs,
        "mt_full_good": "".join(mt76_segs_good), "mt_full_bad": "".join(mt76_segs_bad),
        "mt_segs_good": mt76_segs_good, "mt_segs_bad": mt76_segs_bad
    }

    # Case 77: Cohren Waking (Source 1016)
    # Error: "combat fatigues" -> "pink pajamas"
    src77 = "Silently stepping out of his bunk, he put on his combat fatigues, throwing on his assault vest, and packed up his belongings."
    ref77 = "他默默走下床，穿上作战服，套上冲锋衣，再收拾好储物柜里的东西。"
    src77_segs = [
        "Silently stepping out of his bunk, ",
        "he put on his combat fatigues, throwing on his assault vest, ",
        "and packed up his belongings."
    ]
    ref77_segs = [
        "他默默走下床，",
        "穿上作战服，套上冲锋衣，",
        "再收拾好储物柜里的东西。"
    ]
    mt77_segs_good = ref77_segs
    mt77_segs_bad = [
        "他默默走下床，",
        "穿上他那鲜艳的粉红色睡衣，", # Hallucination/Tone Error
        "再收拾好储物柜里的东西。"
    ]
    cases["Case77"] = {
        "label": "Case 77 (Combat Gear)",
        "src_full": src77, "ref_full": ref77,
        "src_segs": src77_segs, "ref_segs": ref77_segs,
        "mt_full_good": "".join(mt77_segs_good), "mt_full_bad": "".join(mt77_segs_bad),
        "mt_segs_good": mt77_segs_good, "mt_segs_bad": mt77_segs_bad
    }

    # Case 78: FOB Lights (Source 1034)
    # Error: "powered by generators" -> "powered by hamsters"
    src78 = "Cohren and Nemic walked together under the light of the FOB's many lights, which were powered by generators."
    ref78 = "科伦和内米奇并肩行走，他们面前是许多由发电机供电的照明灯。"
    src78_segs = [
        "Cohren and Nemic walked together ",
        "under the light of the FOB's many lights, ",
        "which were powered by generators."
    ]
    ref78_segs = [
        "科伦和内米奇并肩行走，",
        "沐浴在前方作战基地的灯光下，",
        "这些灯是由发电机供电的。"
    ]
    mt78_segs_good = ref78_segs
    mt78_segs_bad = [
        "科伦和内米奇并肩行走，",
        "沐浴在前方作战基地的灯光下，",
        "这些灯是由一群奔跑的仓鼠供电的。" # Absurd Hallucination
    ]
    cases["Case78"] = {
        "label": "Case 78 (FOB Lights)",
        "src_full": src78, "ref_full": ref78,
        "src_segs": src78_segs, "ref_segs": ref78_segs,
        "mt_full_good": "".join(mt78_segs_good), "mt_full_bad": "".join(mt78_segs_bad),
        "mt_segs_good": mt78_segs_good, "mt_segs_bad": mt78_segs_bad
    }

    # Case 79: Support Siege (Source 1049)
    # Error: "flush out insurgency" -> "join the insurgency"
    src79 = "We are to support the ongoing siege and later attack of Ianlos to flush out the insurgency there."
    ref79 = "我们将支持对伊安洛斯的持续围困和后续攻击行动，清剿那里的叛乱分子。"
    src79_segs = [
        "We are to support the ongoing siege ",
        "and later attack of Ianlos ",
        "to flush out the insurgency there."
    ]
    ref79_segs = [
        "我们将支持持续的围困，",
        "以及随后对伊安洛斯的攻击，",
        "以此清剿那里的叛乱分子。"
    ]
    mt79_segs_good = ref79_segs
    mt79_segs_bad = [
        "我们将支持持续的围困，",
        "以及随后对伊安洛斯的攻击，",
        "以此正式加入那里的叛乱分子。" # Action Reversal/Treason
    ]
    cases["Case79"] = {
        "label": "Case 79 (Siege Orders)",
        "src_full": src79, "ref_full": ref79,
        "src_segs": src79_segs, "ref_segs": ref79_segs,
        "mt_full_good": "".join(mt79_segs_good), "mt_full_bad": "".join(mt79_segs_bad),
        "mt_segs_good": mt79_segs_good, "mt_segs_bad": mt79_segs_bad
    }

    # Case 80: Tank Ambush (Source 1070)
    # Error: "taking cover" -> "sunbathing"
    src80 = "Bullets pinged off the tanks side as Shock Troopers leaped off the tanks, taking cover behind them."
    ref80 = "子弹从坦克侧面乒乒乓乓射来，突击队成员纷纷跳下坦克，躲在后面寻找掩护。"
    src80_segs = [
        "Bullets pinged off the tanks side ",
        "as Shock Troopers leaped off the tanks, ",
        "taking cover behind them."
    ]
    ref80_segs = [
        "子弹从坦克侧面乒乒乓乓射来，",
        "突击队成员纷纷跳下坦克，",
        "躲在后面寻找掩护。"
    ]
    mt80_segs_good = ref80_segs
    mt80_segs_bad = [
        "子弹从坦克侧面乒乒乓乓射来，",
        "突击队成员纷纷跳下坦克，",
        "躺在地上享受日光浴。" # Absurd Action Error
    ]
    cases["Case80"] = {
        "label": "Case 80 (Tank Ambush)",
        "src_full": src80, "ref_full": ref80,
        "src_segs": src80_segs, "ref_segs": ref80_segs,
        "mt_full_good": "".join(mt80_segs_good), "mt_full_bad": "".join(mt80_segs_bad),
        "mt_segs_good": mt80_segs_good, "mt_segs_bad": mt80_segs_bad
    }


    """
    Constructed cases from WMT24 en-zh data for XCOMET strategy comparison.
    Selected 20 NEW examples focusing on LONG sentences/paragraphs with MULTIPLE segments (3-4).
    Half of the BAD segments contain obvious semantic/structural errors.
    """
    # Case 81: Crypto status (Source 70)
    # Error: "Nor does... signal anything" -> "strongly implies"
    src81 = "\"Nor does the approval signal anything about the Commission's views as to the status of other crypto assets under the federal securities laws or about the current state of non-compliance of certain crypto asset market participants with the federal securities laws,\" Gensler said."
    ref81 = "对于其他加密资产在联邦证券法下的地位或某些加密资产市场参与者不遵守联邦证券法的现状，这一批准也没有表明美国证监会的任何看法。”詹斯勒表示。"
    src81_segs = [
        "\"Nor does the approval signal anything about the Commission's views ",
        "as to the status of other crypto assets under the federal securities laws ",
        "or about the current state of non-compliance of certain crypto asset market participants with the federal securities laws,\" ",
        "Gensler said."
    ]
    ref81_segs = [
        "这一批准也没有表明美国证监会的任何看法，",
        "对于其他加密资产在联邦证券法下的地位",
        "或某些加密资产市场参与者不遵守联邦证券法的现状，”",
        "詹斯勒表示。"
    ]
    mt81_segs_good = ref81_segs
    mt81_segs_bad = [
        "这一批准强烈暗示了美国证监会的观点，", # Semantic Error (Nor does signal -> Strongly implies)
        "对于其他加密资产在联邦证券法下的地位",
        "以及某些加密资产市场参与者完全遵守联邦证券法的现状，”", # Semantic Error (Non-compliance -> Compliance)
        "詹斯勒表示。"
    ]
    cases["Case81"] = {
        "label": "Case 81 (Crypto Status)",
        "src_full": src81, "ref_full": ref81,
        "src_segs": src81_segs, "ref_segs": ref81_segs,
        "mt_full_good": "".join(mt81_segs_good), "mt_full_bad": "".join(mt81_segs_bad),
        "mt_segs_good": mt81_segs_good, "mt_segs_bad": mt81_segs_bad
    }

    # Case 82: Crypto critique (Source 74)
    # Error: "crashing and burning" -> "thriving"
    src82 = "\"With the flagrantly lawless crypto industry crashing and burning due to a mountain of arrests, criminal convictions, bankruptcies, lawsuits, scandals, massive losses, and millions of investor and customer victims, who would have thought that the SEC would come to its rescue...\""
    ref82 = "“加密货币行业存在肆意逮捕、刑事定罪、破产、诉讼、丑闻、巨额损失等现象，正在崩溃和瓦解，数百万投资者和客户沦为受害者，但谁能想到，美国证监会会出手相救..."
    src82_segs = [
        "\"With the flagrantly lawless crypto industry crashing and burning ",
        "due to a mountain of arrests, criminal convictions, bankruptcies, lawsuits, scandals, massive losses, ",
        "and millions of investor and customer victims, ",
        "who would have thought that the SEC would come to its rescue...\""
    ]
    ref82_segs = [
        "“随着无法无天的加密货币行业正在崩溃和瓦解，",
        "由于大量的逮捕、刑事定罪、破产、诉讼、丑闻和巨额损失，",
        "以及数百万投资者和客户沦为受害者，",
        "谁能想到，美国证监会会出手相救..."
    ]
    mt82_segs_good = ref82_segs
    mt82_segs_bad = [
        "“随着合法合规的加密货币行业蒸蒸日上，", # Semantic Error (Lawless/Crashing -> Lawful/Thriving)
        "得益于大量的逮捕、刑事定罪、破产、诉讼、丑闻和巨额损失，", # Logic Error (Due to negative things -> Thriving)
        "以及数百万投资者和客户沦为受害者，",
        "谁能想到，美国证监会会出手相救..."
    ]
    cases["Case82"] = {
        "label": "Case 82 (Crypto Critique)",
        "src_full": src82, "ref_full": ref82,
        "src_segs": src82_segs, "ref_segs": ref82_segs,
        "mt_full_good": "".join(mt82_segs_good), "mt_full_bad": "".join(mt82_segs_bad),
        "mt_segs_good": mt82_segs_good, "mt_segs_bad": mt82_segs_bad
    }

    # Case 83: Rent stats (Source 100)
    # Error: "double digit increase" -> "decrease"
    src83 = "Every type of property (from one to four bedroom to a room in a property) has had a double digit increase in rents since the Cost of Living legislation was introduced in October 2022, exceeding the annual average increases in rents over the previous 12 years by a factor of at least three."
    ref83 = "自2022年10月生活成本立法出台以来，从一居室到四居室，再到房产中的一个房间，各种类型房产的租金都出现了两位数的增长，较之前12年的年均租金增幅至少高出三倍。"
    src83_segs = [
        "Every type of property has had a double digit increase in rents ",
        "since the Cost of Living legislation was introduced in October 2022, ",
        "exceeding the annual average increases in rents over the previous 12 years ",
        "by a factor of at least three."
    ]
    ref83_segs = [
        "各种类型房产的租金都出现了两位数的增长，",
        "自2022年10月生活成本立法出台以来，",
        "超过了之前12年的年均租金增幅",
        "至少三倍。"
    ]
    mt83_segs_good = ref83_segs
    mt83_segs_bad = [
        "各种类型房产的租金都出现了微不足道的下降，", # Semantic Error (Increase -> Decrease)
        "自2022年10月生活成本立法出台以来，",
        "远低于之前12年的年均租金增幅，", # Semantic Error (Exceeding -> Lower than)
        "至少三倍。" # Contradiction
    ]
    cases["Case83"] = {
        "label": "Case 83 (Rent Stats)",
        "src_full": src83, "ref_full": ref83,
        "src_segs": src83_segs, "ref_segs": ref83_segs,
        "mt_full_good": "".join(mt83_segs_good), "mt_full_bad": "".join(mt83_segs_bad),
        "mt_segs_good": mt83_segs_good, "mt_segs_bad": mt83_segs_bad
    }

    # Case 84: Market Forces (Source 101)
    # Error: "unwillingness to believe" -> "firm belief"
    src84 = "An unwillingness to believe that market forces are what dictates prices rather than government has led to enormous rent price rises which could have been avoided with greater understanding, negotiation and discussion with the sector."
    ref84 = "由于不愿意相信是市场力量而不是政府在决定价格，导致租金价格大幅上涨，而如果能够更多了解这一问题，与有关部门进行更多的谈判和讨论，就可以避免这种情况。"
    src84_segs = [
        "An unwillingness to believe that market forces are what dictates prices rather than government ",
        "has led to enormous rent price rises ",
        "which could have been avoided ",
        "with greater understanding, negotiation and discussion with the sector."
    ]
    ref84_segs = [
        "由于不愿意相信是市场力量而不是政府在决定价格，",
        "导致租金价格大幅上涨，",
        "这本是可以避免的，",
        "如果能与有关部门进行更多的了解、谈判和讨论。"
    ]
    mt84_segs_good = ref84_segs
    mt84_segs_bad = [
        "由于坚信政府而不是市场力量决定价格，", # Misinterpretation
        "导致租金价格大幅下跌，", # Semantic Error (Rises -> Falls)
        "这种情况是不可避免的，", # Semantic Error (Avoided -> Unavoidable)
        "即使与有关部门进行更多的了解、谈判和讨论。"
    ]
    cases["Case84"] = {
        "label": "Case 84 (Market Forces)",
        "src_full": src84, "ref_full": ref84,
        "src_segs": src84_segs, "ref_segs": ref84_segs,
        "mt_full_good": "".join(mt84_segs_good), "mt_full_bad": "".join(mt84_segs_bad),
        "mt_segs_good": mt84_segs_good, "mt_segs_bad": mt84_segs_bad
    }

    # Case 85: Stabilize Rents (Source 108)
    # Error: "encourage investment" -> "ban investment"
    src85 = "To stabilise rental prices in the long term then the government must encourage greater investment and growth in the private rented sector (PRS) while simultaneously funding a substantial growth in the supply of social housing."
    ref85 = "为了长期稳定租金价格，政府必须鼓励私人租赁部门加大投资和促进增长，同时为社会住房供应的大幅增长提供资金。"
    src85_segs = [
        "To stabilise rental prices in the long term ",
        "then the government must encourage greater investment and growth in the private rented sector (PRS) ",
        "while simultaneously funding a substantial growth ",
        "in the supply of social housing."
    ]
    ref85_segs = [
        "为了长期稳定租金价格，",
        "政府必须鼓励私人租赁部门加大投资和促进增长，",
        "同时为大幅增长提供资金",
        "在社会住房供应方面。"
    ]
    mt85_segs_good = ref85_segs
    mt85_segs_bad = [
        "为了长期扰乱租金价格，", # Semantic Error (Stabilize -> Disrupt)
        "政府必须禁止私人租赁部门的投资和增长，", # Semantic Error (Encourage -> Ban)
        "同时为大幅增长提供资金",
        "在社会住房供应方面。"
    ]
    cases["Case85"] = {
        "label": "Case 85 (Stabilize Rents)",
        "src_full": src85, "ref_full": ref85,
        "src_segs": src85_segs, "ref_segs": ref85_segs,
        "mt_full_good": "".join(mt85_segs_good), "mt_full_bad": "".join(mt85_segs_bad),
        "mt_segs_good": mt85_segs_good, "mt_segs_bad": mt85_segs_bad
    }

    # Case 86: Greggs Wages (Source 114)
    # Error: "long time before deflation" -> "immediate deflation"
    src86 = "However, she added that it would be \"a long time before we see deflation\" that would allow the group to start reducing prices, with retailers among those facing higher wage bills due to increases in the national living wage."
    ref86 = "但她补充道，“我们需要很长一段时间才能看到通货紧缩”，届时公司才能开始降价，由于全国最低生活工资上涨，零售商也面临着更高的工资支出。"
    src86_segs = [
        "However, she added that it would be \"a long time before we see deflation\" ",
        "that would allow the group to start reducing prices, ",
        "with retailers among those facing higher wage bills ",
        "due to increases in the national living wage."
    ]
    ref86_segs = [
        "但她补充道，“我们需要很长一段时间才能看到通货紧缩”",
        "届时公司才能开始降价，",
        "零售商也面临着更高的工资支出，",
        "由于全国最低生活工资上涨。"
    ]
    mt86_segs_good = ref86_segs
    mt86_segs_bad = [
        "但她补充道，“我们马上就能看到通货紧缩”", # Semantic Error (Long time -> Immediate)
        "这将迫使集团开始大幅提价，", # Logic Error (Deflation -> Raise prices)
        "零售商也面临着更高的工资支出，",
        "由于全国最低生活工资的降低。" # Semantic Error (Increases -> Decreases)
    ]
    cases["Case86"] = {
        "label": "Case 86 (Greggs Wages)",
        "src_full": src86, "ref_full": ref86,
        "src_segs": src86_segs, "ref_segs": ref86_segs,
        "mt_full_good": "".join(mt86_segs_good), "mt_full_bad": "".join(mt86_segs_bad),
        "mt_segs_good": mt86_segs_good, "mt_segs_bad": mt86_segs_bad
    }

    # Case 87: Greggs Analyst (Source 120)
    # Error: "lead the way" -> "ignored"
    src87 = "He added: \"Festive bakes and chocolate orange muffins lead the way over Christmas, but bears may point to sales growth slowing over the year, and the fourth quarter was the lowest of 2023.\""
    ref87 = "他补充道：“节日烘焙食品和巧克力橙松饼在圣诞节期间广受欢迎，但熊市可能会导致全年销售增长放缓，第四季度销售额为2023年最低水平。”"
    src87_segs = [
        "He added: \"Festive bakes and chocolate orange muffins lead the way over Christmas, ",
        "but bears may point to sales growth slowing over the year, ",
        "and the fourth quarter was the lowest of 2023.\""
    ]
    ref87_segs = [
        "他补充道：“节日烘焙食品和巧克力橙松饼在圣诞节期间广受欢迎，",
        "但看空者可能会指出全年销售增长放缓，",
        "且第四季度为2023年最低水平。”"
    ]
    mt87_segs_good = ref87_segs
    mt87_segs_bad = [
        "他补充道：“节日烘焙食品和巧克力橙松饼在圣诞节期间无人问津，", # Semantic Error (Lead the way -> Ignored)
        "但看多者可能会指出全年销售增长加速，", # Semantic Error (Slowing -> Accelerating)
        "且第四季度为2023年最低水平。”"
    ]
    cases["Case87"] = {
        "label": "Case 87 (Greggs Analyst)",
        "src_full": src87, "ref_full": ref87,
        "src_segs": src87_segs, "ref_segs": ref87_segs,
        "mt_full_good": "".join(mt87_segs_good), "mt_full_bad": "".join(mt87_segs_bad),
        "mt_segs_good": mt87_segs_good, "mt_segs_bad": mt87_segs_bad
    }

    # Case 88: Jail Oversight (Source 141)
    # Error: "collect and report" -> "hide and destroy"
    src88 = "A jail oversight office can also collect and report data, helping administrators, policymakers, and the public understand our jail system and advocate for data-driven solutions to jail deaths."
    ref88 = "监狱监督办公室还可以收集和报告数据，帮助管理者、决策者和公众了解监狱系统，并倡导数据驱动的解决方案，解决监狱死亡率问题。"
    src88_segs = [
        "A jail oversight office can also collect and report data, ",
        "helping administrators, policymakers, and the public understand our jail system ",
        "and advocate for data-driven solutions to jail deaths."
    ]
    ref88_segs = [
        "监狱监督办公室还可以收集和报告数据，",
        "帮助管理者、决策者和公众了解监狱系统，",
        "并倡导数据驱动的解决方案来解决监狱死亡问题。"
    ]
    mt88_segs_good = ref88_segs
    mt88_segs_bad = [
        "监狱监督办公室还可以隐藏和销毁数据，", # Semantic Error (Collect/Report -> Hide/Destroy)
        "帮助管理者、决策者和公众了解监狱系统，",
        "并阻碍数据驱动的解决方案，导致更多监狱死亡。" # Semantic Error (Advocate/Solutions -> Hinder/Deaths)
    ]
    cases["Case88"] = {
        "label": "Case 88 (Jail Oversight)",
        "src_full": src88, "ref_full": ref88,
        "src_segs": src88_segs, "ref_segs": ref88_segs,
        "mt_full_good": "".join(mt88_segs_good), "mt_full_bad": "".join(mt88_segs_bad),
        "mt_segs_good": mt88_segs_good, "mt_segs_bad": mt88_segs_bad
    }

    # Case 89: Boeing Grounding (Source 146)
    # Error: "grounded" -> "flying"
    src89 = "But with the majority of Boeing 737 MAX 9 jets grounded around the country after an Alaska Airlines fuselage blowout on Jan. 6, some prospective passengers may want to know how to tell what type of plane they'll be on."
    ref89 = "1月6日，阿拉斯加航空公司的一架波音737 Max 9客机发生内嵌舱门掉落事故后，美国大部分波音737 Max 9客机停飞，一些潜在乘客可能想知道如何辨别他们将乘坐的飞机类型。"
    src89_segs = [
        "But with the majority of Boeing 737 MAX 9 jets grounded around the country ",
        "after an Alaska Airlines fuselage blowout on Jan. 6, ",
        "some prospective passengers may want to know how to tell what type of plane they'll be on."
    ]
    ref89_segs = [
        "但随着大多数波音737 MAX 9客机在全国范围内停飞，",
        "在1月6日阿拉斯加航空公司发生机身爆裂事故后，",
        "一些潜在乘客可能想知道如何辨别他们将乘坐的飞机类型。"
    ]
    mt89_segs_good = ref89_segs
    mt89_segs_bad = [
        "但随着大多数波音737 MAX 9客机在全国范围内复飞，", # Semantic Error (Grounded -> Flying again)
        "在1月6日阿拉斯加航空公司发生机身爆炸事故前，", # Time Error (After -> Before)
        "一些潜在乘客可能想知道如何辨别他们将乘坐的飞机类型。"
    ]
    cases["Case89"] = {
        "label": "Case 89 (Boeing Grounding)",
        "src_full": src89, "ref_full": ref89,
        "src_segs": src89_segs, "ref_segs": ref89_segs,
        "mt_full_good": "".join(mt89_segs_good), "mt_full_bad": "".join(mt89_segs_bad),
        "mt_segs_good": mt89_segs_good, "mt_segs_bad": mt89_segs_bad
    }

    # Case 90: Airbus A310 (Source 153)
    # Error: "1983" -> "2023"
    src90 = "The Airbus A310, which was introduced in 1983, has the highest rate of hull losses - 2.53 per 1 million departures - among the models that are still in service as passenger aircraft."
    ref90 = "空客A310于1983年推出，在目前仍在服役的客机型号中，其机身损失事故率最高，每100万次飞行为2.53次。"
    src90_segs = [
        "The Airbus A310, which was introduced in 1983, ",
        "has the highest rate of hull losses - 2.53 per 1 million departures - ",
        "among the models that are still in service as passenger aircraft."
    ]
    ref90_segs = [
        "空客A310于1983年推出，",
        "其机身损失事故率最高 - 每100万次飞行为2.53次 - ",
        "在目前仍在服役的客机型号中。"
    ]
    mt90_segs_good = ref90_segs
    mt90_segs_bad = [
        "空客A310于2023年推出，", # Date Error
        "其机身损失事故率最低 - 每100万次飞行为0次 - ", # Semantic Error (Highest -> Lowest)
        "在目前仍在服役的客机型号中。"
    ]
    cases["Case90"] = {
        "label": "Case 90 (Airbus A310)",
        "src_full": src90, "ref_full": ref90,
        "src_segs": src90_segs, "ref_segs": ref90_segs,
        "mt_full_good": "".join(mt90_segs_good), "mt_full_bad": "".join(mt90_segs_bad),
        "mt_segs_good": mt90_segs_good, "mt_segs_bad": mt90_segs_bad
    }

    # Case 91: Betta Edu Backlash (Source 160)
    # Error: "backlash" -> "support"
    src91 = "The minister has faced significant public backlash following the exposure of a leaked memo where she purportedly instructed Oluwatoyin Madein, the Accountant-General (AG) of the federation, to transfer N585 million to a private account."
    ref91 = "一份泄露的备忘录显示，埃杜指示尼日利亚联邦总会计师Oluwatoyin Madein将5.85亿奈拉转入一个私人账户，此举遭到了公众的强烈反对。"
    src91_segs = [
        "The minister has faced significant public backlash ",
        "following the exposure of a leaked memo ",
        "where she purportedly instructed Oluwatoyin Madein... to transfer N585 million to a private account."
    ]
    ref91_segs = [
        "这位部长遭到了公众的强烈反对，",
        "在一份泄露的备忘录曝光后，",
        "她在备忘录中指示Oluwatoyin Madein将5.85亿奈拉转入一个私人账户。"
    ]
    mt91_segs_good = ref91_segs
    mt91_segs_bad = [
        "这位部长受到了公众的热烈支持，", # Semantic Error (Backlash -> Support)
        "在一份泄露的备忘录曝光后，",
        "她在备忘录中指示将5.85亿奈拉捐给慈善机构。" # Hallucination
    ]
    cases["Case91"] = {
        "label": "Case 91 (Betta Edu)",
        "src_full": src91, "ref_full": ref91,
        "src_segs": src91_segs, "ref_segs": ref91_segs,
        "mt_full_good": "".join(mt91_segs_good), "mt_full_bad": "".join(mt91_segs_bad),
        "mt_segs_good": mt91_segs_good, "mt_segs_bad": mt91_segs_bad
    }

    # Case 92: Bank Official (Source 164)
    # Error: "incorrect" -> "correct"
    src92 = "He added that unless she can provide a valid reason for such approval, it may not necessarily involve fraudulent intent, but it is nonetheless incorrect, and certainty is lacking."
    ref92 = "该法律专员补充道，除非埃杜能提出有关批准的正当理由，否则该行为即使不一定涉及欺诈意图，但也是不正确的，而且缺乏确定性。"
    src92_segs = [
        "He added that unless she can provide a valid reason for such approval, ",
        "it may not necessarily involve fraudulent intent, ",
        "but it is nonetheless incorrect, and certainty is lacking."
    ]
    ref92_segs = [
        "他补充道，除非她能为这种批准提供正当理由，",
        "这不一定涉及欺诈意图，",
        "但尽管如此也是不正确的，且缺乏确定性。"
    ]
    mt92_segs_good = ref92_segs
    mt92_segs_bad = [
        "他补充道，即使她提供了批准的正当理由，", # Logic Error
        "这也肯定涉及欺诈意图，", # Semantic Error (Not necessarily -> Definitely)
        "而且是完全正确和确定的。" # Semantic Error (Incorrect -> Correct)
    ]
    cases["Case92"] = {
        "label": "Case 92 (Bank Official)",
        "src_full": src92, "ref_full": ref92,
        "src_segs": src92_segs, "ref_segs": ref92_segs,
        "mt_full_good": "".join(mt92_segs_good), "mt_full_bad": "".join(mt92_segs_bad),
        "mt_segs_good": mt92_segs_good, "mt_segs_bad": mt92_segs_bad
    }

    # Case 93: Celeste Pride (Source 170)
    # Error: "proud" -> "ashamed"
    src93 = "Celeste is so proud of its tiny parent game it throws in references like the memorial, the \"2000 M... 2500 M...\" progression, the power-up, the flag at the top."
    ref93 = "《Celeste》对自己的亲子小游戏非常自豪，在游戏中加入了纪念物、“2000米……2500米……”进程、强化道具、顶部旗帜等元素。"
    src93_segs = [
        "Celeste is so proud of its tiny parent game ",
        "it throws in references like the memorial, ",
        "the \"2000 M... 2500 M...\" progression, the power-up, the flag at the top."
    ]
    ref93_segs = [
        "《Celeste》对自己的亲子小游戏非常自豪，",
        "它在游戏中加入了纪念物、",
        "“2000米……2500米……”进程、强化道具、顶部旗帜等元素。"
    ]
    mt93_segs_good = ref93_segs
    mt93_segs_bad = [
        "《Celeste》对自己的亲子小游戏感到非常羞耻，", # Semantic Error (Proud -> Ashamed)
        "它删除了所有相关的纪念物、", # Semantic Error (Throws in -> Removes)
        "“2000米……2500米……”进程、强化道具、顶部旗帜等元素。"
    ]
    cases["Case93"] = {
        "label": "Case 93 (Celeste Pride)",
        "src_full": src93, "ref_full": ref93,
        "src_segs": src93_segs, "ref_segs": ref93_segs,
        "mt_full_good": "".join(mt93_segs_good), "mt_full_bad": "".join(mt93_segs_bad),
        "mt_segs_good": mt93_segs_good, "mt_segs_bad": mt93_segs_bad
    }

    # Case 94: Celeste Incarnation (Source 173)
    # Error: "favorite" -> "hated"
    src94 = "I think that would've been my favorite incarnation as like, an emotional arc... as is, the main single player arc of the commercial game is ultimately a small piece of it."
    ref94 = "我认为，这是我最喜欢的化身，就像情感弧线……事实上，在商业游戏中，主要单人模式故事线最终只是其中的一小部分。"
    src94_segs = [
        "I think that would've been my favorite incarnation as like, an emotional arc... ",
        "as is, the main single player arc of the commercial game ",
        "is ultimately a small piece of it."
    ]
    ref94_segs = [
        "我认为那会是我最喜欢的化身，就像情感弧线……",
        "事实上，商业游戏的主要单人模式故事线",
        "最终只是其中的一小部分。"
    ]
    mt94_segs_good = ref94_segs
    mt94_segs_bad = [
        "我认为那是我最讨厌的化身，没有任何情感弧线……", # Semantic Error (Favorite -> Hated)
        "事实上，商业游戏的主要单人模式故事线",
        "是其全部内容，占据了100%的比例。" # Semantic Error (Small piece -> All)
    ]
    cases["Case94"] = {
        "label": "Case 94 (Celeste Incarnation)",
        "src_full": src94, "ref_full": ref94,
        "src_segs": src94_segs, "ref_segs": ref94_segs,
        "mt_full_good": "".join(mt94_segs_good), "mt_full_bad": "".join(mt94_segs_bad),
        "mt_segs_good": mt94_segs_good, "mt_segs_bad": mt94_segs_bad
    }

    # Case 95: Brain Sides (Source 238)
    # Error: "both sides" -> "one side"
    src95 = "But when using both the creative (i.e., guitar, painting, etc.) and analytical sides of my brain, it feels as though I increased my mental capacity — making me better in both spaces."
    ref95 = "但是，当我同时使用大脑的创造功能（如弹吉他、绘画等）和分析功能时，感觉自己的脑容量有所增加，让我在这两个领域都变得更好。"
    src95_segs = [
        "But when using both the creative (i.e., guitar, painting, etc.) and analytical sides of my brain, ",
        "it feels as though I increased my mental capacity ",
        "— making me better in both spaces."
    ]
    ref95_segs = [
        "但是，当我同时使用大脑的创造功能和分析功能时，",
        "感觉自己的脑容量有所增加，",
        "让我在这两个领域都变得更好。"
    ]
    mt95_segs_good = ref95_segs
    mt95_segs_bad = [
        "但是，当我只使用大脑的创造功能而忽略分析功能时，", # Semantic Error (Both -> One)
        "感觉自己的脑容量明显萎缩，", # Semantic Error (Increased -> Shrunk)
        "让我在这两个领域都变得更糟。" # Semantic Error (Better -> Worse)
    ]
    cases["Case95"] = {
        "label": "Case 95 (Brain Sides)",
        "src_full": src95, "ref_full": ref95,
        "src_segs": src95_segs, "ref_segs": ref95_segs,
        "mt_full_good": "".join(mt95_segs_good), "mt_full_bad": "".join(mt95_segs_bad),
        "mt_segs_good": mt95_segs_good, "mt_segs_bad": mt95_segs_bad
    }

    # Case 96: Political Roles (Source 254)
    # Error: "like" -> "hate"
    src96 = "And i do like characters having long conversations, superbly aware of their political roles and yet ignorant to their emotional needs. Part of growing up."
    ref96 = "我也喜欢人物长时间的对话，他们非常清楚自己的政治角色，但对情感需求一无所知。这是成长的一部分吧。"
    src96_segs = [
        "And i do like characters having long conversations, ",
        "superbly aware of their political roles ",
        "and yet ignorant to their emotional needs. ",
        "Part of growing up."
    ]
    ref96_segs = [
        "我也喜欢人物长时间的对话，",
        "他们非常清楚自己的政治角色，",
        "但对情感需求一无所知。",
        "这是成长的一部分吧。"
    ]
    mt96_segs_good = ref96_segs
    mt96_segs_bad = [
        "我讨厌人物长时间的对话，", # Semantic Error (Like -> Hate)
        "他们对自己的政治角色一无所知，", # Semantic Error (Aware -> Ignorant)
        "但对情感需求了如指掌。", # Semantic Error (Ignorant -> Aware)
        "这是成长的一部分吧。"
    ]
    cases["Case96"] = {
        "label": "Case 96 (Political Roles)",
        "src_full": src96, "ref_full": ref96,
        "src_segs": src96_segs, "ref_segs": ref96_segs,
        "mt_full_good": "".join(mt96_segs_good), "mt_full_bad": "".join(mt96_segs_bad),
        "mt_segs_good": mt96_segs_good, "mt_segs_bad": mt96_segs_bad
    }

    # Case 97: Introvert Advertising (Source 304)
    # Error: "Advertise here" -> "Do not advertise"
    src97 = "Advertise here, follow the algorithm, share your personality. Get a blue tick on that service, be available 24/7, answer inquiries with a video of yourself, respond immediately."
    ref97 = "在这里做广告，要遵循算法，分享个性。勾选该服务，全天候可用，用自己的视频回答询问，并立即做出回应。"
    src97_segs = [
        "Advertise here, follow the algorithm, share your personality. ",
        "Get a blue tick on that service, be available 24/7, ",
        "answer inquiries with a video of yourself, respond immediately."
    ]
    ref97_segs = [
        "在这里做广告，遵循算法，分享个性。",
        "在该服务上获得蓝勾认证，全天候待命，",
        "用自己的视频回答询问，立即回应。"
    ]
    mt97_segs_good = ref97_segs
    mt97_segs_bad = [
        "不要在这里做广告，无视算法，隐藏个性。", # Semantic Error (Opposite instructions)
        "在该服务上获得蓝勾认证，全天候待命，",
        "永远不要回答询问，或者用假视频回应。" # Semantic Error (Opposite/Hallucination)
    ]
    cases["Case97"] = {
        "label": "Case 97 (Introvert Ads)",
        "src_full": src97, "ref_full": ref97,
        "src_segs": src97_segs, "ref_segs": ref97_segs,
        "mt_full_good": "".join(mt97_segs_good), "mt_full_bad": "".join(mt97_segs_bad),
        "mt_segs_good": mt97_segs_good, "mt_segs_bad": mt97_segs_bad
    }

    # Case 98: Air Purifier (Source 323)
    # Error: "uncomfortably" -> "comfortable"
    src98 = "It feels uncomfortably passive aggressive turning up my air purifier to max, but I can smell the wood dust, so it would be justified even if he weren't an anti-masker."
    ref98 = "我把空气净化器调到最大档，仍然感到难受；我闻到木屑的味道，即使他不反对戴口罩，也是有道理的。"
    src98_segs = [
        "It feels uncomfortably passive aggressive turning up my air purifier to max, ",
        "but I can smell the wood dust, ",
        "so it would be justified even if he weren't an anti-masker."
    ]
    ref98_segs = [
        "把空气净化器调到最大档感觉有点消极攻击的不适感，",
        "但我能闻到木屑味，",
        "所以即使他不是反口罩人士，这也是合理的。"
    ]
    mt98_segs_good = ref98_segs
    mt98_segs_bad = [
        "把空气净化器关掉感觉很舒服，", # Semantic Error (Max -> Off; Uncomfortable -> Comfortable)
        "但我闻到花香的味道，", # Hallucination (Wood dust -> Flowers)
        "所以这是毫无道理的，即使他戴了口罩。" # Semantic Error (Justified -> Unjustified)
    ]
    cases["Case98"] = {
        "label": "Case 98 (Air Purifier)",
        "src_full": src98, "ref_full": ref98,
        "src_segs": src98_segs, "ref_segs": ref98_segs,
        "mt_full_good": "".join(mt98_segs_good), "mt_full_bad": "".join(mt98_segs_bad),
        "mt_segs_good": mt98_segs_good, "mt_segs_bad": mt98_segs_bad
    }

    # Case 99: HTML Stream (Source 402)
    # Error: "out of order" -> "in order"
    src99 = "It would be streamed out of order, but the browser would assemble the HTML document as if it were streamed in order."
    ref99 = "虽然流式传输的顺序是乱的，但浏览器会将HTML文档按照流式顺序排列。"
    src99_segs = [
        "It would be streamed out of order, ",
        "but the browser would assemble the HTML document ",
        "as if it were streamed in order."
    ]
    ref99_segs = [
        "它将乱序流式传输，",
        "但浏览器会组装HTML文档，",
        "就像它是按顺序流式传输一样。"
    ]
    mt99_segs_good = ref99_segs
    mt99_segs_bad = [
        "它将严格按顺序流式传输，", # Semantic Error (Out of order -> In order)
        "但浏览器会将HTML文档撕碎，", # Hallucination (Assemble -> Tear apart)
        "就像它从未被流式传输过一样。" # Semantic Error
    ]
    cases["Case99"] = {
        "label": "Case 99 (HTML Stream)",
        "src_full": src99, "ref_full": ref99,
        "src_segs": src99_segs, "ref_segs": ref99_segs,
        "mt_full_good": "".join(mt99_segs_good), "mt_full_bad": "".join(mt99_segs_bad),
        "mt_segs_good": mt99_segs_good, "mt_segs_bad": mt99_segs_bad
    }

    # Case 100: Congressman Votes (Source 445)
    # Error: "explained" -> "hid"
    src100 = "So that kind of famously as a Congressman explained to all of your votes on Facebook, which is a rare concession by authority to say, okay, this is why I did what I did."
    ref100 = "众所周知，您作为国会议员，在Facebook上说明了所有投票的理由；而当权者很少会做出这样的让步，说，“好吧，这就是我做那些事的原因。”"
    src100_segs = [
        "So that kind of famously as a Congressman explained to all of your votes on Facebook, ",
        "which is a rare concession by authority to say, ",
        "okay, this is why I did what I did."
    ]
    ref100_segs = [
        "众所周知，你作为国会议员在Facebook上解释了所有的投票，",
        "这是当权者罕见的让步，即",
        "好吧，这就是我这么做的原因。"
    ]
    mt100_segs_good = ref100_segs
    mt100_segs_bad = [
        "众所周知，您作为国会议员，在Facebook上隐瞒了所有投票理由，", # Semantic Error (Explained -> Hid)
        "这是当权者常见的傲慢表现，说，", # Semantic Error (Rare concession -> Common arrogance)
        "“闭嘴，我不需要解释我做了什么。”" # Hallucination/Opposite
    ]
    cases["Case100"] = {
        "label": "Case 100 (Congressman Votes)",
        "src_full": src100, "ref_full": ref100,
        "src_segs": src100_segs, "ref_segs": ref100_segs,
        "mt_full_good": "".join(mt100_segs_good), "mt_full_bad": "".join(mt100_segs_bad),
        "mt_segs_good": mt100_segs_good, "mt_segs_bad": mt100_segs_bad
    }

    return cases