
"""
Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
Selected 20 New examples where errors are confined to a single segment 
and exhibit obvious semantic/structural deviations.
"""
from typing import Dict, Any, List

def build_wmt23_cases() -> Dict[str, Dict[str, Any]]:
    cases = {}

    def add_case(label: str, src_index: int, src_full: str, 
                 segs_good: List[str], segs_bad: List[str]):
        """
        Helper to construct a case object ensuring strict data consistency.
        ref_full is automatically generated from joining segs_good.
        """
        cases[label] = {
            "label": label,
            "source_index": src_index,
            "src_full": src_full,
            "ref_full": "".join(segs_good),
            "mt_full_good": "".join(segs_good),
            "mt_full_bad": "".join(segs_bad),
            "mt_segs_good": segs_good,
            "mt_segs_bad": segs_bad
        }

    # Case 1: Product Manual (Error in Segment 2)
    add_case(
        "Case 1 (Instruction Manual)",
        382,
        "使用产品前请仔细阅读本说明书，并妥善保管。",
        ["Before using the product, please carefully read this manual", " and treat it with care."],
        ["Before using the product, please carefully read this manual", " and throw it away immediately."]
    )

    # Case 2: Headset Usage (Error in Segment 1)
    add_case(
        "Case 2 (Device Usage)",
        386,
        "通话时，将耳机麦克风朝向嘴部可使通话更清晰。",
        ["For crisper calls, angle the headset microphone toward your mouth", " when speaking on the phone."],
        ["For muffled noise, cover the headset microphone with tape", " when speaking on the phone."]
    )

    # Case 3: Maintenance Timing (Error in Segment 2)
    add_case(
        "Case 3 (Maintenance)",
        1297,
        "上午收到机油，下午就换上了。",
        ["Changed the oil in the afternoon,", " after receiving it in the morning."],
        ["Changed the oil in the afternoon,", " before ordering it in the evening."]
    )

    # Case 4: Product Feedback (Error in Segment 2)
    add_case(
        "Case 4 (Customer Review - Oil)",
        1303,
        "快递飞速，包装没有挑剔的地方，用了这个油，发动机声音明显下降。",
        ["Express shipping is quick, and packing requirements are flexible,", " the engine noise has greatly decreased after using this oil."],
        ["Express shipping is quick, and packing requirements are flexible,", " the engine noise has become unbearably loud after using this oil."]
    )

    # Case 5: Toothpaste Review (Error in Segment 1)
    add_case(
        "Case 5 (Product Review - Toothpaste)",
        1364,
        "膏体里带颗粒的，清爽感恰到好处，用后牙齿爽滑，上比某二手东强太多啦，那里说是进口的东西，可好多东西没有中文标签，不知从哪里批发的。",
        ["The paste has granules, feels perfectly refreshing, and leaves teeth with a smooth finish after usage, it is superior to a used item by a wide margin,", " although it claims to be imported, several items lack Chinese labelling, I don't know where they are purchased in wholesale."],
        ["The paste is gritty, feels painful, and destroys teeth enamel after usage, it is inferior to a used item,", " although it claims to be imported, several items lack Chinese labelling, I don't know where they are purchased in wholesale."]
    )

    # Case 6: Child Preference (Error in Segment 1)
    add_case(
        "Case 6 (User Preference)",
        1512,
        "小孩很不喜欢牙膏的味道，每次都说辣，也许用别的词形容才更贴切，小孩的词库里暂时只有这个词。",
        ["The kid doesn't particularly enjoy the flavor of toothpaste, and always claims it's spicy,", " perhaps a different word would be more suited to describe it, currently the child's vocabulary consists of just this one word."],
        ["The kid absolutely loves the flavor of toothpaste, and always claims it's delicious,", " perhaps a different word would be more suited to describe it, currently the child's vocabulary consists of just this one word."]
    )

    # Case 7: Price Fluctuation (Error in Segment 1)
    add_case(
        "Case 7 (Pricing)",
        1528,
        "降价太厉害了，一百多入现在99！",
        ["More than 100 yuan is now only 99,", " a too-strong price reduction!"],
        ["More than 100 yuan has increased to 999,", " a too-strong price reduction!"]
    )

    # Case 8: Mattress Thickness (Error in Segment 1)
    add_case(
        "Case 8 (Product Dimension)",
        1580,
        "34CM的床垫不是一般的厚，不要床直接睡床垫都可以了。",
        ["Since the mattress is only 34CM thick,", " you can sleep on it alone without a bed."],
        ["Since the mattress is microscopic,", " you can sleep on it alone without a bed."]
    )

    # Case 9: Jewelry Review (Error in Segment 1)
    add_case(
        "Case 9 (Jewelry)",
        1610,
        "小猴子可爱，亚马逊优惠力度真大，周生生官网直接发货，前天晚上下单，今天下午就收到了，真快。",
        ["The small monkey is adorable, and Amazon has excellent savings,", " the products are shipped directly from the Chow Sang Sang official website, I ordered it the previous evening, and I got it this afternoon, Really quickly."],
        ["The small monkey is hideous, and Amazon has terrible prices,", " the products are shipped directly from the Chow Sang Sang official website, I ordered it the previous evening, and I got it this afternoon, Really quickly."]
    )

    # Case 10: Thermal Insulation (Error in Segment 1)
    add_case(
        "Case 10 (Product Performance)",
        1639,
        "宝宝出生一周购买，保温超强。",
        ["The insulation is very sturdy", " and was bought a week after the baby was delivered."],
        ["The insulation is non-existent", " and was bought a week after the baby was delivered."]
    )

    # Case 11: Shipping Speed (Error in Segment 2)
    add_case(
        "Case 11 (Logistics Speed)",
        1645,
        "9月2号下单，预计到货是21号，结果12号就到了，提前了差不多10天，速度是真快！",
        ["My order, which I placed on September 2, was supposed to arrive on the 21st, but it did so on the 12th instead—nearly 10 days earlier—", "so the speed is really quick!"],
        ["My order, which I placed on September 2, was supposed to arrive on the 21st, but it did so on the 12th instead—nearly 10 days earlier—", "so the speed is painfully slow!"]
    )

    # Case 12: Audio Quality (Error in Segment 2)
    add_case(
        "Case 12 (Audio Quality)",
        1742,
        "我惊愕了，音质竟然是环绕3D立体！",
        ["I was astounded to discover that", " the sound quality was surround 3D!"],
        ["I was astounded to discover that", " the sound quality was mono and static!"]
    )

    # Case 13: Delivery Service (Error in Segment 1)
    add_case(
        "Case 13 (Service Quality)",
        1841,
        "快递服务非常方便安全，送货师傅对客户非常负责，每次都会准时提醒去快递柜取货。",
        ["the rapid delivery service is really convenient and secure,", " and the delivery master is highly mindful to consumers, He will remind you to pick up the things from the express cabinet on schedule every time."],
        ["the rapid delivery service is dangerous and inconvenient,", " and the delivery master is highly mindful to consumers, He will remind you to pick up the things from the express cabinet on schedule every time."]
    )

    # Case 14: Material Breathability (Error in Segment 2)
    add_case(
        "Case 14 (Material Property)",
        1860,
        "透气性也很好。",
        ["Breathability is", " excellent as well."],
        ["Breathability is", " non-existent as well."]
    )

    # Case 15: Clothing Quality (Error in Segment 2)
    add_case(
        "Case 15 (Clothing Quality)",
        1885,
        "衣服的质量非常好。",
        ["The clothing is", " of excellent quality."],
        ["The clothing is", " of garbage quality."]
    )

    # Case 16: Toy Review (Error in Segment 1)
    add_case(
        "Case 16 (Toy Quality)",
        1904,
        "很好玩，组件精度很高，做工精致，质量跟乐高差不多了，做活动时价格十分优惠。",
        ["It is a lot of fun, the parts are extremely precise,", " the craftsmanship is excellent, the quality is comparable to Lego, and the price is extremely advantageous when engaging in activities."],
        ["It is boring and tedious, the parts are sloppy,", " the craftsmanship is excellent, the quality is comparable to Lego, and the price is extremely advantageous when engaging in activities."]
    )

    # Case 17: Financial Product Launch (Error in Segment 2)
    add_case(
        "Case 17 (Financial News)",
        523,
        "个人养老金理财产品终于上线了。",
        ["Items for overseeing individual benefits riches", " have at long last been presented."],
        ["Items for overseeing individual benefits riches", " have been permanently banned."]
    )

    # Case 18: Economic Performance (Error in Segment 1)
    add_case(
        "Case 18 (Economic Analysis)",
        597,
        "专家分析认为，台湾经济开年表现欠佳，增长压力不小。",
        ["Agreeing to master examination, Taiwan's economy performed ineffectively at the starting of the year,", " and the development weight isn't little."],
        ["Agreeing to master examination, Taiwan's economy performed magnificently at the starting of the year,", " and the development weight isn't little."]
    )

    # Case 19: Economic Resilience (Error in Segment 1)
    add_case(
        "Case 19 (Macroeconomics)",
        619,
        "“应对新冠疫情冲击，中国经济展现出了强大韧性。",
        ["\"The Chinese economy has illustrated exceptional versatility", " within the confront of the impacts of the unused crown plague."],
        ["\"The Chinese economy has shown zero resilience", " within the confront of the impacts of the unused crown plague."]
    )

    # Case 20: Rescue Operations (Error in Segment 2)
    add_case(
        "Case 20 (Disaster Relief)",
        732,
        "中国救援队在土耳其持续开展救援行动 已营救4名被困人员",
        ["The Chinese rescue team continues to carry out rescue operations in Turkey", " and has rescued 4 trapped people"],
        ["The Chinese rescue team continues to carry out rescue operations in Turkey", " and has harmed 4 trapped people"]
    )

    """
    Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
    Selected 20 New examples where errors are confined to a single segment 
    and exhibit obvious semantic/structural deviations.
    """

    # Case 21: Drug Report (Error in Segment 2)
    add_case(
        "Case 21 (Drug Analysis)",
        306,
        "早在2012年，对医院、监狱和其他机构人员的调查就已经揭示了甲基苯丙胺在伊拉克的重要性。",
        ["Examinations of staff individuals in clinics, correctional facilities, and other teach as early as 2012", " had demonstrated the importance of methamphetamine in Iraq."],
        ["Examinations of staff individuals in clinics, correctional facilities, and other teach as early as 2012", " had proven that methamphetamine is unknown in Iraq."]
    )

    # Case 22: Drug Statistics (Error in Segment 1)
    add_case(
        "Case 22 (Statistics)",
        351,
        "据估计，2018年使用大麻的有 1.92 亿人，使之成为全球使用最多的毒品。",
        ["Cannabis was the top substance used globally in 2018", " by 0.192 billion people."],
        ["Cannabis was the least used substance globally in 2018", " by 0.192 billion people."]
    )

    # Case 23: Product Name (Error in Segment 2)
    add_case(
        "Case 23 (Product Info)",
        433,
        "产品名称 :Redmi AirDots 2 真无线蓝牙耳机",
        ["Product name: Redmi AirDots 2", " Wireless Headphones"],
        ["Product name: Redmi AirDots 2", " Wired Landline Telephone"]
    )

    # Case 24: Safety Warning (Error in Segment 1)
    add_case(
        "Case 24 (Safety Instruction)",
        448,
        "1 请勿以任何理由拆开.维修或改装耳机, 否则可能会导致起火甚至彻底损坏本产品;",
        ["Do not alter or remove the headset,", " it may cause fire or damage."],
        ["Please freely alter or remove the headset,", " it may cause fire or damage."]
    )

    # Case 25: EU Summit (Error in Segment 2)
    add_case(
        "Case 25 (Political News)",
        488,
        "欧盟峰会为难民政策争吵不休，欧盟官员：接收过程每一步都存在瓶颈",
        ["Conflicts over refugee policy at the EU summit amongst EU officials;", " obstacles at every stage of the admission process"],
        ["Conflicts over refugee policy at the EU summit amongst EU officials;", " smooth sailing at every stage of the admission process"]
    )

    # Case 26: Gym Business (Error in Segment 1)
    add_case(
        "Case 26 (Business Analysis)",
        556,
        "从市场前景看，健身房当然算得上是一门有“钱途”的好生意。",
        ["Of course, gyms might be seen as a good business with \"money potential\"", " from the perspective of market prospects."],
        ["Of course, gyms are seen as a bankrupt business with zero potential", " from the perspective of market prospects."]
    )

    # Case 27: Service Quality (Error in Segment 2)
    add_case(
        "Case 27 (Service Strategy)",
        566,
        "首先，要靠服务留住用户。",
        ["To keep users,", " it is first vital to rely on services."],
        ["To keep users,", " it is first vital to ignore services completely."]
    )

    # Case 28: Real Estate Market (Error in Segment 1)
    add_case(
        "Case 28 (Real Estate)",
        591,
        "“楼市在回暖，此时更需要一针‘强心剂’。",
        ["\"The genuine domain advertise is growing,", " but right presently we require a \"heart booster\" shot."],
        ["\"The genuine domain advertise is collapsing,", " but right presently we require a \"heart booster\" shot."]
    )

    # Case 29: Economic Data (Error in Segment 1)
    add_case(
        "Case 29 (Trade Statistics)",
        604,
        "统计显示，1月台湾地区进出口双降，出超金额仅23.4亿美元，创近3年来最低水平。",
        ["Concurring to measurements, Taiwan's imports and trades diminished in January,", " with the excess coming to a record-low US$2.34 billion."],
        ["Concurring to measurements, Taiwan's imports and trades skyrocketed in January,", " with the excess coming to a record-low US$2.34 billion."]
    )

    # Case 30: Economic Potential (Error in Segment 2)
    add_case(
        "Case 30 (Global Economy)",
        633,
        "中国经济发展潜力不断释放，将给世界各国带来更多利好，成为全球复苏的重要引擎。",
        ["Ceaselessly discharging China's financial advancement potential will advantage more countries around the world", " and serve as a key catalyst for the world economy to restore."],
        ["Ceaselessly discharging China's financial advancement potential will advantage more countries around the world", " and serve as a key obstacle for the world economy to restore."]
    )

    # Case 31: Student Statistics (Error in Segment 1)
    add_case(
        "Case 31 (Education)",
        641,
        "与2021年相比，2022年应届留学生规模同比增长8.6%，回国求职留学生数量再创新高。",
        ["The number of universal understudies graduating in 2022 will rise by 8.6% yearly compared to 2021,", " and a record-breaking number of worldwide understudies will return to China to explore for work."],
        ["The number of universal understudies graduating in 2022 will drop to zero compared to 2021,", " and a record-breaking number of worldwide understudies will return to China to explore for work."]
    )

    # Case 32: Medical News (Error in Segment 2)
    add_case(
        "Case 32 (Healthcare)",
        658,
        "央视网消息：种植牙是很多缺牙患者的理想选择，但一直以来，动辄上万元一颗的价格让很多患者望而却步。",
        ["CCTV news: For numerous patients who are lost teeth, dental inserts are the most excellent alternative,", " however, for an awfully long time, the taken a toll of tens of thousands of dollars per embed has hindered numerous individuals."],
        ["CCTV news: For numerous patients who are lost teeth, dental inserts are the most excellent alternative,", " however, for an awfully long time, the free cost of zero dollars per embed has hindered numerous individuals."]
    )

    # Case 33: Infrastructure (Error in Segment 1)
    add_case(
        "Case 33 (City Planning)",
        674,
        "打造城市级高品质无线局域网。",
        ["Construct a top-notch remote LAN", " at the city level."],
        ["Destroy the existing remote LAN", " at the city level."]
    )

    # Case 34: Disaster Relief (Error in Segment 2)
    add_case(
        "Case 34 (Rescue Operations)",
        736,
        "救援队将根据灾区实际和救援现场情况，科学调整救援力量，持续深入开展救援工作。",
        ["The rescue team will adjust the rescue forces based on the actual situation in the disaster area and the conditions at the rescue site,", " and will continue to carry out rescue operations in a comprehensive and thorough manner."],
        ["The rescue team will adjust the rescue forces based on the actual situation in the disaster area and the conditions at the rescue site,", " and will stop all rescue operations immediately."]
    )

    # Case 35: Earthquake History (Error in Segment 1)
    add_case(
        "Case 35 (Historical Event)",
        826,
        "1976年7月28日，中国河北省唐山市和丰南县在16小时内发生两次7级以上强烈地震。",
        ["On July 28, 1976, two strong earthquakes with a magnitude of 7 or above occurred within 16 hours", " in Tangshan City and Fengnan County, Hebei Province, China."],
        ["On July 28, 1976, two gentle breezes occurred within 16 hours", " in Tangshan City and Fengnan County, Hebei Province, China."]
    )

    # Case 36: International Relations (Error in Segment 2)
    add_case(
        "Case 36 (Tourism Policy)",
        957,
        "新加坡政府和旅游业对中国市场寄予厚望。",
        ["The Singapore government and tourism industry", " have high hopes for the Chinese market."],
        ["The Singapore government and tourism industry", " have completely banned the Chinese market."]
    )

    # Case 37: Clothing Review (Error in Segment 2)
    add_case(
        "Case 37 (Customer Complaint)",
        1252,
        "这件衣服当时我们购买的时候是一件拉链帽衫，最后送给我确是一件短袖。",
        ["When we purchased this dress, it had a zipper hood,", " but I ended up wearing a short-sleeved version."],
        ["When we purchased this dress, it had a zipper hood,", " but I ended up wearing a space suit."]
    )

    # Case 38: Shopping Experience (Error in Segment 2)
    add_case(
        "Case 38 (Service Review)",
        1260,
        "买衣服容易，退货也容易。",
        ["Both clothing purchases,", " and returns are simple."],
        ["Both clothing purchases,", " and returns are impossible."]
    )

    # Case 39: Product Authenticity (Error in Segment 1)
    add_case(
        "Case 39 (Fake Goods)",
        1344,
        "亚马逊卖假货，我也是服了。",
        ["Amazon sells counterfeit goods,", " and I'm confident of it."],
        ["Amazon sells authentic goods,", " and I'm confident of it."]
    )

    # Case 40: Customer Satisfaction (Error in Segment 2)
    add_case(
        "Case 40 (User Feedback)",
        1740,
        "买之前心里真的没底，直到打开一看包装就感觉很正！",
        ["I was pretty apprehensive before I bought it,", " but after I cracked open the box, I was certain!"],
        ["I was pretty apprehensive before I bought it,", " but after I cracked open the box, I was devastated!"]
    )
    """
    Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
    Selected 20 New examples where errors are confined to a single segment 
    and exhibit obvious semantic/structural deviations.
    Batch 3: Cases 41-60.
    """
    # Case 41: Document Info (Error in Segment 2)
    add_case(
        "Case 41 (Technical Standard)",
        2,
        "增加了本文件适用对象(见第 1 章)，",
        ["The related objects of this record have been included", " (see Chapter 1),"],
        ["The related objects of this record have been included", " (see the Trash Can),"]
    )

    # Case 42: Headset Usage (Error in Segment 1)
    add_case(
        "Case 42 (Instruction Manual)",
        384,
        "01 佩戴耳机",
        ["putting on", " headphones"],
        ["taking off", " headphones"]
    )

    # Case 43: Wearing Instructions (Error in Segment 2)
    add_case(
        "Case 43 (Device Usage)",
        385,
        "将耳帐斜向下轻轻塞入耳道，以轻摆头部耳机不田动为家。",
        ["Gently insert the ear tents downwardly into the ear canal,", " then gently swing your head to move the earphones."],
        ["Gently insert the ear tents downwardly into the ear canal,", " then gently smash your head to break the earphones."]
    )

    # Case 44: Charging Instructions (Error in Segment 1)
    add_case(
        "Case 44 (Product Guide)",
        388,
        "充电开机使用前，请先撕掉左右耳机充各点处的网几，将充电盒及耳机充满电。",
        ["Please remove the nets from the left and right earphone charging points before charging and using,", " and charge the charging box and headphones completely before using."],
        ["Please glue the nets to the left and right earphone charging points before charging and using,", " and charge the charging box and headphones completely before using."]
    )

    # Case 45: Charging Setup (Error in Segment 2)
    add_case(
        "Case 45 (Device Charging)",
        389,
        "插入充电线可同时给耳机和充电盒充电。",
        ["To charge the headset and the charging case simultaneously,", " plug in the charging cord."],
        ["To charge the headset and the charging case simultaneously,", " cut the charging cord."]
    )

    # Case 46: User Review Intro (Error in Segment 1)
    add_case(
        "Case 46 (Customer Feedback)",
        1257,
        "这是我的第一个长评，希望这不是我的最后一个。",
        ["My first lengthy review,", " which I hope won't be my last."],
        ["My final suicide note,", " which I hope won't be my last."]
    )

    # Case 47: Clothing Material (Error in Segment 2)
    add_case(
        "Case 47 (Product Review - Pants)",
        1267,
        "毛裤的做工、颜色、手感都不错，柔软/薄厚适中，没有煤油味儿。",
        ["The wool pants have good construction, a nice color, a decent amount of softness/thinness,", " and no kerosene odor."],
        ["The wool pants have good construction, a nice color, a decent amount of softness/thinness,", " and a strong smell of rotting fish."]
    )

    # Case 48: Camera Review (Error in Segment 1)
    add_case(
        "Case 48 (Electronic Product)",
        1291,
        "换货挺迅速，晚上有点模糊，视角不是很广，但是普通家用够了。",
        ["Although the viewing angle is not very wide, the replacement is extremely quick, and it is a little hazy at night;", " it is sufficient for everyday home use."],
        ["Although the viewing angle is infinite, the replacement took ten years, and it is crystal clear at night;", " it is sufficient for everyday home use."]
    )

    # Case 49: Store Trust (Error in Segment 2)
    add_case(
        "Case 49 (Brand Trust)",
        1299,
        "亚马逊自营商品还是很放心的。",
        ["Self-operated Amazon goods", " continue to be quite comforting."],
        ["Self-operated Amazon goods", " continue to be a total scam."]
    )

    # Case 50: Oil Price (Error in Segment 1)
    add_case(
        "Case 50 (Product Value)",
        1304,
        "性价比不错，二百以下很难找到合成机油。",
        ["The price-to-performance ratio is favorable,", " and finding synthetic motor oil for under 200 is challenging."],
        ["The price-to-performance ratio is terrible,", " and finding synthetic motor oil for under 200 is challenging."]
    )

    # Case 51: Product Support (Error in Segment 2)
    add_case(
        "Case 51 (Brand Loyalty)",
        1305,
        "支持星冠，支持胜牌，对车有效保护。",
        ["Support the winning card, the star crown,", " and successfully defend the vehicle."],
        ["Support the winning card, the star crown,", " and successfully destroy the vehicle."]
    )

    # Case 52: Return Policy (Error in Segment 1)
    add_case(
        "Case 52 (Customer Service)",
        1311,
        "1，你退货，买家承担所有运费。",
        ["1, You return the merchandise,", " and the buyer is responsible for all shipping charges."],
        ["1, You keep the merchandise,", " and the buyer is responsible for all shipping charges."]
    )

    # Case 53: Future Purchase (Error in Segment 1)
    add_case(
        "Case 53 (Customer Intent)",
        1326,
        "以后应该不会继续购买了！",
        ["Should not make", " further purchases in the future!"],
        ["Must make", " further purchases in the future!"]
    )

    # Case 54: Packaging Issue (Error in Segment 2)
    add_case(
        "Case 54 (Shipping Damage)",
        1329,
        "就一个压变形的箱子。",
        ["Just a box", " that is crumpled and distorted."],
        ["Just a box", " that is pristine and perfect."]
    )

    # Case 55: Pricing & Warranty (Error in Segment 1)
    add_case(
        "Case 55 (Value Comparison)",
        1332,
        "宁愿国内贵一半，有保障点。",
        ["I'd rather pay half as much in China", " and get a warranty."],
        ["I'd rather pay ten times more in Mars", " and get a warranty."]
    )

    # Case 56: Product Color (Error in Segment 2)
    add_case(
        "Case 56 (User Preference)",
        1336,
        "整个杯子的颜色是我狠喜欢的，橘红色，虽然我家是男宝，但是还是觉得橘红配黄色很亮眼。",
        ["The color of the entire cup is my favorite, orange red, and even though my family is all boys,", " I believe orange red and yellow are extremely appealing."],
        ["The color of the entire cup is my favorite, orange red, and even though my family is all boys,", " I believe orange red and yellow are hideous and disgusting."]
    )

    # Case 57: Product Size (Error in Segment 1)
    add_case(
        "Case 57 (Product Dimensions)",
        1337,
        "比我想象的小一些，不过本来就是拿来带出门用的，所以还可以接受。",
        ["It's smaller than I expected, but it's designed to be removed,", " so it's fine."],
        ["It's larger than a planet, and it's designed to be permanent,", " so it's fine."]
    )

    # Case 58: Bag Design (Error in Segment 1)
    add_case(
        "Case 58 (Product Utility)",
        1341,
        "喜欢它轻便，开口大，就是两个独立的分隔袋对我来说有点多余，感觉不知道装什么好。",
        ["I like how light it is, and how spacious the entrance is,", " but two distinct compartments are a bit unnecessary for me, and I'm not sure what to put into it."],
        ["I hate how heavy it is, and how narrow the entrance is,", " but two distinct compartments are a bit unnecessary for me, and I'm not sure what to put into it."]
    )

    # Case 59: Smell Description (Error in Segment 2)
    add_case(
        "Case 59 (Sensory Experience)",
        1353,
        "回想起有点像发廊里劣质烫发水的味道。",
        ["Now that I think about it,", " the hair salon kind of smells like terrible perm water."],
        ["Now that I think about it,", " the hair salon kind of smells like delicious French perfume."]
    )

    # Case 60: General Verdict (Error in Segment 2)
    add_case(
        "Case 60 (Final Review)",
        1359,
        "反正我觉得这个东西确实不错。",
        ["Whatever the case,", " I think this is a really great thing."],
        ["Whatever the case,", " I think this is a really terrible thing."]
    )

    """
    Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
    Selected 20 New examples where errors are confined to a single segment 
    and exhibit obvious semantic/structural deviations.
    Batch 4: Cases 61-80.
    """
    # Case 61: Book Reading Level (Error in Segment 2)
    add_case(
        "Case 61 (Book Review)",
        1467,
        "适合粗读,不能细看。",
        ["Suitable for rough reading,", " not for detailed perusal."],
        ["Suitable for rough reading,", " not for cooking dinner."]
    )

    # Case 62: Book Critique (Error in Segment 1)
    add_case(
        "Case 62 (Content Quality)",
        1468,
        "本书的逻辑性很好,但是内容不够丰满。",
        ["The logic of this book is good,", " but the content is not substantial enough."],
        ["The logic of this book is terrible,", " but the content is not substantial enough."]
    )

    # Case 63: Book Structure (Error in Segment 2)
    add_case(
        "Case 63 (Writing Style)",
        1470,
        "另外,本书感觉像是拼凑而成。",
        ["In addition, the book feels", " like it was pieced together."],
        ["In addition, the book feels", " like a masterpiece of art."]
    )

    # Case 64: Writing Difficulty (Error in Segment 1)
    add_case(
        "Case 64 (Writing Analysis)",
        1482,
        "写高大空很容易，难的是写的通俗易懂。",
        ["Writing high-sounding emptiness is easy,", " the difficulty lies in writing clearly and simply."],
        ["Writing high-sounding emptiness is impossible,", " the difficulty lies in writing clearly and simply."]
    )

    # Case 65: E-commerce Tricks (Error in Segment 2)
    add_case(
        "Case 65 (Shopping Advice)",
        1509,
        "电商套路深，有时间还是去超市买了。",
        ["E-commerce tricks are deep,", " better to go to the supermarket if you have time."],
        ["E-commerce tricks are deep,", " better to fly to the moon if you have time."]
    )

    # Case 66: Stock Status (Error in Segment 1)
    add_case(
        "Case 66 (Product Availability)",
        1517,
        "是正品，只是没货了，不知道什么时候再上架。",
        ["It is authentic, but out of stock,", " don't know when it will be available again."],
        ["It is fake, but out of stock,", " don't know when it will be available again."]
    )

    # Case 67: Cable Quality (Error in Segment 2)
    add_case(
        "Case 67 (Product Review - Cable)",
        1532,
        "低价好线，不发烧就这个够了！",
        ["Good low-priced cable,", " if you're not an audiophile this is enough!"],
        ["Good low-priced cable,", " if you're not an alien this is enough!"]
    )

    # Case 68: Beginner Choice (Error in Segment 1)
    add_case(
        "Case 68 (User Recommendation)",
        1604,
        "还行，作为新手来说，值得入手",
        ["It's okay, for a beginner,", " it is worth buying"],
        ["It's trash, for a beginner,", " it is worth buying"]
    )

    # Case 69: Courier Responsibility (Error in Segment 2)
    add_case(
        "Case 69 (Service Compliment)",
        1611,
        "邮政快递员也很负责。",
        ["The postal courier", " is also very responsible."],
        ["The postal courier", " is also very irresponsible."]
    )

    # Case 70: Product Design (Error in Segment 1)
    add_case(
        "Case 70 (Design Aesthetics)",
        1627,
        "这是一款设计风格非常简洁的产品。",
        ["This is a product", " with a very simple design style."],
        ["This is a banana", " with a very simple design style."]
    )

    # Case 71: Design Outcome (Error in Segment 2)
    add_case(
        "Case 71 (Design Failure)",
        1630,
        "可惜最终效果没有能够实现。",
        ["Unfortunately the final effect", " could not be achieved."],
        ["Unfortunately the final effect", " was achieved perfectly."]
    )

    # Case 72: Empathy for Designer (Error in Segment 1)
    add_case(
        "Case 72 (Emotional Reaction)",
        1635,
        "有点心疼那个产品设计师。",
        ["Feel a bit sorry", " for that product designer."],
        ["Feel a bit hungry", " for that product designer."]
    )

    # Case 73: Return Policy (Error in Segment 2)
    add_case(
        "Case 73 (Service Uniqueness)",
        1700,
        "不提倡退货，但是这样的服务，在中国仅此一家。",
        ["Do not advocate returns, but such service", " is the only one in China."],
        ["Do not advocate returns, but such service", " is strictly prohibited in China."]
    )

    # Case 74: Brand Slogan (Error in Segment 1)
    add_case(
        "Case 74 (Marketing Slogan)",
        1729,
        "梵盾吉普服饰，因为年轻，所以自信！",
        ["Fandun Jeep Clothing,", " because it is young, it is confident!"],
        ["Fandun Jeep Clothing,", " because it is old, it is fearful!"]
    )

    # Case 75: DIY Haircut (Error in Segment 2)
    add_case(
        "Case 75 (Product Benefit)",
        1805,
        "3、完全可以DIY理发，省钱了。",
        ["3. Can completely DIY haircut,", " saved money."],
        ["3. Can completely DIY haircut,", " lost all money."]
    )

    # Case 76: Purchase Summary (Error in Segment 1)
    add_case(
        "Case 76 (Shopping Verdict)",
        1806,
        "综上，非常成功的一次购买。",
        ["In summary,", " a very successful purchase."],
        ["In summary,", " a very disastrous purchase."]
    )

    # Case 77: Family Preference (Error in Segment 2)
    add_case(
        "Case 77 (Platform Loyalty)",
        1842,
        "总之，我们全家都喜欢在亚马逊海外购shopping!",
        ["In short, our whole family loves", " shopping on Amazon Overseas Buy!"],
        ["In short, our whole family loves", " starving on Amazon Overseas Buy!"]
    )

    # Case 78: Packaging Quality (Error in Segment 1)
    add_case(
        "Case 78 (Packaging Review)",
        1646,
        "包装箱不大，直接用的是Vitamix 的原包装，亚马逊也不怕漂洋过海的摔坏了！",
        ["The box is not big, directly using Vitamix original packaging,", " Amazon is not afraid of it breaking across the ocean!"],
        ["The box is gigantic, directly using Vitamix original packaging,", " Amazon is not afraid of it breaking across the ocean!"]
    )

    # Case 79: Delivery Request (Error in Segment 2)
    add_case(
        "Case 79 (Delivery Scheduling)",
        1870,
        "本人要求周六、周日送货！",
        ["I request delivery", " on Saturday and Sunday!"],
        ["I request delivery", " on Mars and Jupiter!"]
    )

    # Case 80: Customer Lament (Error in Segment 1)
    add_case(
        "Case 80 (Brand Relationship)",
        1873,
        "亚马逊，我如何再爱你？",
        ["Amazon,", " how can I love you again?"],
        ["Google,", " how can I love you again?"]
    )

    """
    Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
    Selected 20 NEW examples focusing on LONGER sentences with three segments
    where errors are confined to 1 or 2 segments
    and exhibit obvious semantic/structural deviations.
    No duplicates from previous batches.
    Batch 5: Cases 81-100.
    """
    # Case 81: Pandemic Crisis (Error in Segment 1 and 2)
    add_case(
        "Case 81 (Global Crisis)",
        238,
        "全球性的新冠肺炎疫情已使全世界陷入前所未有的危机。",
        ["The globe is right now encountering", " an unparalleled emergency", " as a result of the COVID-19 scourge."],
        ["The globe is right now enjoying", " a fantastic party", " as a result of the COVID-19 scourge."]
    )

    # Case 82: Healthcare Collapse (Error in Segment 1)
    add_case(
        "Case 82 (Health System)",
        239,
        "在许多国家，这一疾病己使医疗服务机构穷于应付，几近崩溃，各国政府实施的限制措施已在全球经济中造成了严重紊乱。",
        ["The affliction has disabled wellbeing frameworks in a few countries to the point of collapse,", " and government confinements have", " seriously disturbed the world economy."],
        ["The affliction has strengthened wellbeing frameworks in a few countries to the point of perfection,", " and government confinements have", " seriously disturbed the world economy."]
    )

    # Case 83: Drug Market Impact (Error in Segment 3)
    add_case(
        "Case 83 (Market Uncertainty)",
        240,
        "对于毒品市场，这场疫情的影响尚不清楚，也很难预测，但有可能影响深远。",
        ["The outbreak's impacts on the pharma industry are vague", " and difficult to estimate,", " but they may be critical."],
        ["The outbreak's impacts on the pharma industry are vague", " and difficult to estimate,", " but they are absolutely zero."]
    )

    # Case 84: Drug Production (Error in Segment 1 and 3)
    add_case(
        "Case 84 (Supply Chain)",
        241,
        "一些生产者可能会被迫寻找新的方法来制造毒品，因为限制流动的同时也限制了前体和基本化学品的获取。",
        ["Due to the constrained get to to antecedents and essential chemicals", " caused by development confinements,", " certain makers may be obliged to create modern strategies of medicine generation."],
        ["Due to the unlimited access to antecedents and essential chemicals", " caused by development confinements,", " certain makers may be obliged to stop all medicine generation."]
    )

    # Case 85: Drug Usage Shift (Error in Segment 2 and 3)
    add_case(
        "Case 85 (Usage Patterns)",
        245,
        "2008年经济危机后，一些吸毒者开始寻找更便宜的合成物质，使用模式转向注射毒品。",
        ["A few sedate clients begun searching for less costly manufactured drugs", " amid the 2008 budgetary emergency,", " and their drug-using propensities changed to sedate infusion."],
        ["A few sedate clients begun searching for less costly manufactured drugs", " amid the 2008 budgetary boom,", " and their drug-using propensities changed to drinking water."]
    )

    # Case 86: Government Intervention (Error in Segment 2)
    add_case(
        "Case 86 (Policy Impact)",
        247,
        "如果各国政府以同样的方式应对当前的经济衰退，预防吸毒和相关风险行为等干预措施和戒毒治疗服务可能会受到沉重打击。",
        ["Intercessions like medicate utilize and related hazard behavior anticipation", " and medicate treatment programs may be extremely hurt", " in the event that governments reacted to the show retreat within the same way."],
        ["Intercessions like medicate utilize and related hazard behavior anticipation", " and medicate treatment programs may be hugely boosted", " in the event that governments reacted to the show retreat within the same way."]
    )

    # Case 87: Vulnerable Populations (Error in Segment 1)
    add_case(
        "Case 87 (Social Issues)",
        253,
        "由于不断上升的失业率和缺乏机会，贫困和弱势人群将更有可能以有害的方式吸毒，患上吸毒病症，并转向与毒品有关的非法活动一无论是生产还是运输。",
        ["The destitute and powerless will be more slanted to utilize drugs hurtfully,", " create sedate utilize disarranges, and lock in in unlawful drug-related activities", "—whether generation or transportation—as a result of expanded unemployment and a need of openings."],
        ["The wealthy and powerful will be more slanted to utilize drugs hurtfully,", " create sedate utilize disarranges, and lock in in unlawful drug-related activities", "—whether generation or transportation—as a result of expanded unemployment and a need of openings."]
    )

    # Case 88: Global Statistics (Error in Segment 2)
    add_case(
        "Case 88 (Drug Statistics)",
        260,
        "2009年估计吸毒者有2.1亿人，占全球15-64岁人口的4.8%,而2018年估计吸毒者有2.69亿人，占这类人口的5.3%。",
        ["In 2018, there were an estimated 0.269 billion drug users,", " which is 5.3% more than there were in 2009,", " when there were an estimated 0.210 billion users."],
        ["In 2018, there were an estimated 0.269 billion drug users,", " which is 5.3% less than there were in 2009,", " when there were an estimated 0.210 billion users."]
    )

    # Case 89: Developing Nations (Error in Segment 1)
    add_case(
        "Case 89 (Regional Growth)",
        261,
        "在过去二十年期间，发展中国家吸毒情况的增长速度远远快于发达国家。",
        ["Sedate utilization has expanded altogether more rapidly", " in developing countries than in industrialized countries", " over the past 20 a long time."],
        ["Sedate utilization has decreased altogether more rapidly", " in developing countries than in industrialized countries", " over the past 20 a long time."]
    )

    # Case 90: Urbanization (Error in Segment 3)
    add_case(
        "Case 90 (Urbanization)",
        267,
        "吸毒情况总体增多的部分原因是人口从农村向城镇的大规模流动一目前全世界人口有一半以上生活在城市地区，而1960年这一比例为34%。",
        ["More than half of the world's populace presently lives in urban locales, up from 34% in 1960,", " and this colossal relocation from rustic to urban regions", " is to a great extent to fault for the common rise in sedate utilize."],
        ["More than half of the world's populace presently lives in urban locales, up from 34% in 1960,", " and this colossal relocation from rustic to urban regions", " has nothing to do with the common rise in sedate utilize."]
    )

    # Case 91: Unemployment Impact (Error in Segment 1)
    add_case(
        "Case 91 (Labor Market)",
        276,
        "例如，劳动力市场的变化，如失业率上升，过去一直与吸毒情况的增加有关，而这场疫情己经迫使全球数千万人失业。",
        ["For occasion, the scourge has tossed tens of millions of individuals out of work universally", " and changes within the labor advertise, such as rising unemployment,", " have already been related to expanded sedate utilize."],
        ["For occasion, the scourge has employed tens of millions of individuals universally", " and changes within the labor advertise, such as rising unemployment,", " have already been related to expanded sedate utilize."]
    )

    # Case 92: Fentanyl Usage (Error in Segment 1)
    add_case(
        "Case 92 (Substance Analysis)",
        288,
        "在北美，芬太尼有的用作海洛因和其他毒品（包括可卡因和甲基苯丙胺）的掺杂剂，有的用来制造假冒的药用类阿片。",
        ["Fentanyl is utilized in North America to create fake pharmaceutical painkillers", " or as a doping specialist for heroin", " and other substances like cocaine and methamphetamine."],
        ["Fentanyl is utilized in North America to create real pharmaceutical painkillers", " or as a doping specialist for heroin", " and other substances like cocaine and methamphetamine."]
    )

    # Case 93: Price Fluctuations (Error in Segment 1 and 2)
    add_case(
        "Case 93 (Market Prices)",
        297,
        "来自墨西哥的证据表明，这已经成为 现实：据报告，2020年3月，从东亚进口的甲基苯丙胺前体短缺促使墨西哥和美国的甲基苯丙胺价格上涨。",
        ["Information from Mexico infers that this is often as of now the case:", " in Walk 2020, Mexico and the Joined together States detailed cost rises for methamphetamine", " due to deficiencies of forerunners provided from East Asia."],
        ["Information from Mexico infers that this is never the case:", " in Walk 2020, Mexico and the Joined together States detailed cost drops for methamphetamine", " due to deficiencies of forerunners provided from East Asia."]
    )

    # Case 94: Opioid Seizures (Error in Segment 1 and 3)
    add_case(
        "Case 94 (Law Enforcement)",
        301,
        "俄罗斯联邦当局截获的阿片剂数量下降了约80%，同时因使用类阿片而接受治疗的人数大幅减少。",
        ["Whereas less individuals are getting treatment for opioid utilization,", " the number of sedatives reallocated by Russian League specialists", " has diminished by generally 80%."],
        ["Whereas more individuals are getting treatment for opioid utilization,", " the number of sedatives reallocated by Russian League specialists", " has increased by generally 80%."]
    )

    # Case 95: Cannabis Legalization (Error in Segment 1)
    add_case(
        "Case 95 (Legal Trends)",
        317,
        "在其中多数法域，大麻使用自合法化以来有所增多，尽管在其他未将非医疗使用大麻合法化的法域也观察到了同样的趋势。",
        ["The larger part of these locales have seen a rise in cannabis utilize since legalization,", " be that as it may, other purviews without such enactment", " have moreover seen this design."],
        ["The larger part of these locales have seen a drop in cannabis utilize since legalization,", " be that as it may, other purviews without such enactment", " have moreover seen this design."]
    )

    # Case 96: Disability Years (Error in Segment 3)
    add_case(
        "Case 96 (Health Impact)",
        340,
        "2007年至2018年间，归因于吸毒的全球残疾调整寿命年数增加了 17%。",
        ["Between 2007 and 2018,", " the number of worldwide disability-adjusted life a long time credited to medicate utilization", " rose by 17%."],
        ["Between 2007 and 2018,", " the number of worldwide disability-adjusted life a long time credited to medicate utilization", " fell by 17%."]
    )

    # Case 97: Tramadol Seizures (Error in Segment 3)
    add_case(
        "Case 97 (Drug Control)",
        363,
        "2017年，全球截获的曲马多数量显著增加，达到125吨以上的峰值。",
        ["In 2017,", " the amount of tramadol that was seized internationally", " peaked at more than 125 tons."],
        ["In 2017,", " the amount of tramadol that was seized internationally", " peaked at zero tons."]
    )

    # Case 98: Drug Deaths (Error in Segment 2)
    add_case(
        "Case 98 (Mortality Stats)",
        376,
        "2017年，因吸毒死亡的约有 58.5万人，其中一半死于丙型肝炎引起的肝病，而注射吸毒者的丙型肝炎大多仍未得到治疗。",
        ["Half of the estimated 0.585 million drug-related deaths in 2017", " were attributable to liver illness brought on by hepatitis C,", " which is largely untreated among injecting drug users."],
        ["Half of the estimated 0.585 million drug-related deaths in 2017", " were attributable to eating too many vegetables,", " which is largely untreated among injecting drug users."]
    )

    # Case 99: Product Warranty (Error in Segment 1)
    add_case(
        "Case 99 (Warranty Policy)",
        470,
        "1 自您签收次日起7 日内, 本产品出现《Redmi AirDots 真无线蓝牙耳机产品性能故障表》所列性能故障的情况, 经由小米售后服务中心检测确定, 可免费享受退货或换货服务;",
        ["1 Task incomplete, Xiaomi allows free return or exchange in 7 days", " for confirmed performance issues", " on \"Redmi AirDots True Wireless Bluetooth Headset Product Performance Failure Table\""],
        ["1 Task incomplete, Xiaomi bans return or exchange in 7 days", " for confirmed performance issues", " on \"Redmi AirDots True Wireless Bluetooth Headset Product Performance Failure Table\""]
    )

    # Case 100: EU Funding (Error in Segment 1)
    add_case(
        "Case 100 (Political Funding)",
        491,
        "根据峰会文件，欧盟领导人呼吁委员会“立即调动大量欧盟资金”，以基础设施和监视等手段加强外部边界。",
        ["EU leaders requested the Commission to \"quickly mobilize substantial EU funding,\"", " including through infrastructure and monitoring, to bolster external borders,", " according to the summit document."],
        ["EU leaders requested the Commission to \"quickly hide substantial EU funding,\"", " including through infrastructure and monitoring, to bolster external borders,", " according to the summit document."]
    )

    """
    Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
    Selected 20 NEW examples focusing on MEDIUM-LENGTH sentences with three segments
    where errors are confined to 1 or 2 segments
    and exhibit obvious semantic/structural deviations.
    No duplicates from previous batches.
    Batch 6: Cases 101-120.
    """
    # Case 101: Political Biography (Error in Segment 1)
    add_case(
        "Case 101 (Biography)",
        1115,
        "莫德罗是自由德国青年运动的一名工作人员，几乎所有东德年轻人都加入了该运动，并在 1973 年至 1989 年期间通过执政的社会统一党 (SED) 晋升为德累斯顿地区党的主席。",
        ["Modrow was a staff member of the Free German Youth Movement,", " which almost all young East Germans joined,", " and rose through the ranks of the ruling Social Unity Party (SED) to become president of the Dresden Regional Party from 1973 to 1989."],
        ["Modrow was a fierce enemy of the Free German Youth Movement,", " which almost all young East Germans joined,", " and rose through the ranks of the ruling Social Unity Party (SED) to become president of the Dresden Regional Party from 1973 to 1989."]
    )

    # Case 102: International Relations (Error in Segment 2)
    add_case(
        "Case 102 (Geopolitics)",
        504,
        "无论是拉拢韩国建立“技术同盟”，还是与韩国、日本和中国台湾地区建立芯片“四方联盟”，亦或借由美日印澳“四边机制”...",
        ["Whether it be to convince South Korea to create a \"specialized organization together,\",", " or to make a chip \"quadrate alliance\" with South Korea, Japan, and Taiwan,", " or to set up a semiconductor industry chain through components such as the \"four-party component\"..."],
        ["Whether it be to convince South Korea to create a \"specialized organization together,\",", " or to declare total war on South Korea, Japan, and Taiwan,", " or to set up a semiconductor industry chain through components such as the \"four-party component\"..."]
    )

    # Case 103: Product Material (Error in Segment 3)
    add_case(
        "Case 103 (Product Review - Leather)",
        1259,
        "说是牛皮的，也确实比较硬，但仔细看的话，局部有很多细小的裂痕，搞不清楚是否是真皮的。",
        ["Although it is stated to be made of cowhide and is in fact fairly hard,", " if you look closely, you will notice that there are numerous tiny fractures in the surrounding area,", " making it unclear whether the leather is real."],
        ["Although it is stated to be made of cowhide and is in fact fairly hard,", " if you look closely, you will notice that there are numerous tiny fractures in the surrounding area,", " making it perfectly clear that the leather is made of pure diamond."]
    )

    # Case 104: Technical Standard (Error in Segment 1)
    add_case(
        "Case 104 (Document Scope)",
        155,
        "本文件收录了 2013 年中华人民共和国国务院批准发布的 4 通用规范汉字表》 中的全部 8105 个规范汉字,它们的代码位置和字形见表1。",
        ["This archive incorporates all 8,105 standardized Chinese characters", " from the 2013 discharge of the People's Republic of China's State Council's Fourth Common Standardized Chinese Character List,", " their code areas and glyphs are recorded in Table 1."],
        ["This archive permanently deletes all 8,105 standardized Chinese characters", " from the 2013 discharge of the People's Republic of China's State Council's Fourth Common Standardized Chinese Character List,", " their code areas and glyphs are recorded in Table 1."]
    )

    # Case 105: Cosmetic Scent (Error in Segment 2)
    add_case(
        "Case 105 (User Experience)",
        1351,
        "用第二片的时候选的毛孔细致的，但是一打开就一股怪味，以为是正常的，因为没用过这款。",
        ["The pores on the second piece were fine when I used it,", " but there was an odd smell when I opened it,", " I hadn't used this one before, so I assumed it was typical."],
        ["The pores on the second piece were fine when I used it,", " but there was a delightful fragrance of fresh roses when I opened it,", " I hadn't used this one before, so I assumed it was typical."]
    )

    # Case 106: Customer Service (Error in Segment 2)
    add_case(
        "Case 106 (Service Interaction)",
        1748,
        "无奈之下，打亚马逊客服，客服告诉了卖家的手机号码，我再打电话问店家。",
        ["I dialed Amazon customer support out of desperation,", " the seller's mobile phone number was provided by customer support,", " I dialed the shop once more."],
        ["I dialed Amazon customer support out of desperation,", " the customer support screamed at me and hung up immediately,", " I dialed the shop once more."]
    )

    # Case 107: Diaper Quality (Error in Segment 3)
    add_case(
        "Case 107 (Product Material)",
        1322,
        "这是第二次从亚马逊上购买帮宝适尿布，选择了欧洲进口纸张的尿布。",
        ["I've purchased Pampers diapers from Amazon twice already,", " my baby was 2 months old when I purchased a box of diapers,", " I decided on diapers made of imported European paper."],
        ["I've purchased Pampers diapers from Amazon twice already,", " my baby was 2 months old when I purchased a box of diapers,", " I decided on diapers made of recycled sandpaper."]
    )

    # Case 108: Manufacturing Process (Error in Segment 2)
    add_case(
        "Case 108 (Industrial Process)",
        511,
        "据专业人士估算，基于专业化分工，半导体产品的整个生产过程需要跨越各国边境70次以上，全程达到100天。",
        ["Using specialized division of labor,", " experts estimate that the complete production process for semiconductor devices requires crossing international borders more than 70 times", " and takes 100 days."],
        ["Using specialized division of labor,", " experts estimate that the complete production process for semiconductor devices requires staying inside a single room forever", " and takes 100 days."]
    )

    # Case 109: Packaging Complaint (Error in Segment 2)
    add_case(
        "Case 109 (Packaging)",
        1902,
        "再说一下包装，就是用保鲜膜包了几层，里面连个包装塑料泡泡纸都没有。",
        ["Let's talk about the packaging once more,", " it's just wrapped in plastic wrap for a few layers,", " not even a package of plastic bubble wrap inside."],
        ["Let's talk about the packaging once more,", " it is encased in a solid gold chest with diamond locks,", " not even a package of plastic bubble wrap inside."]
    )

    # Case 110: Corporate Layoffs (Error in Segment 3)
    add_case(
        "Case 110 (Business News)",
        506,
        "最近，美国知名芯片制造设备供应商泛林集团宣布，受最新一轮对华芯片制造设备出口禁令影响，集团将解雇1300名全职员工。",
        ["The recent round of export restrictions on chip manufacturing equipment to China have caused the Lam Group,", " a well-known supplier of chip manufacturing equipment in the United States,", " to announce that it will be forced to lay off 1,300 full-time employees."],
        ["The recent round of export restrictions on chip manufacturing equipment to China have caused the Lam Group,", " a well-known supplier of chip manufacturing equipment in the United States,", " to announce that it will hire 1 billion new employees immediately."]
    )

    # Case 111: Inventory Issue (Error in Segment 2)
    add_case(
        "Case 111 (Shopping Issue)",
        1253,
        "收到衣服后立即与亚马逊客服取得了联系，当时确认衣服还是有库存的，我要求换货。",
        ["I immediately contacted Amazon customer support after getting the garments,", " it was determined at that time that there was still inventory of the clothing,", " I made a replacement request."],
        ["I immediately contacted Amazon customer support after getting the garments,", " it was determined at that time that the clothing never existed in this universe,", " I made a replacement request."]
    )

    # Case 112: Economic Data (Error in Segment 3)
    add_case(
        "Case 112 (Trade Statistics)",
        598,
        "据台当局财政事务主管部门公布，台湾地区今年1月出口315.1亿美元，年减21.2%，连续第5个月负增长。",
        ["Concurring to the monetary undertakings office of the Taiwan specialists,", " Taiwan's trades in January this year were 31.51 billion US dollars, an yearly diminish of 21.2%,", " and it was the fifth continuous month of negative development."],
        ["Concurring to the monetary undertakings office of the Taiwan specialists,", " Taiwan's trades in January this year were 31.51 billion US dollars, an yearly diminish of 21.2%,", " and it was the fifth continuous month of explosive growth."]
    )

    # Case 113: Product Sentiment (Error in Segment 1)
    add_case(
        "Case 113 (Product Feedback)",
        1623,
        "因为看点评里说的都很好，所有对这款锅很有期待，拿到手的时候也很喜欢，很轻巧。",
        ["Everyone is anticipating this pot because of the positive comments,", " When I first got it, I really liked it,", " although it is quite lightweight."],
        ["Everyone is hating this pot because of the terrible comments,", " When I first got it, I really liked it,", " although it is quite lightweight."]
    )

    # Case 114: Logistics Failure (Error in Segment 3)
    add_case(
        "Case 114 (Delivery Issue)",
        1519,
        "黑五的时候下的单，结果等了好久都没收到，后面一查询，包裹早已被签收，问了身边的同事都说没有帮签收。",
        ["I ordered on Black Friday, but despite waiting a long time, I never got the item,", " the box had already been signed for when I checked it later,", " my coworkers denied signing for it when I questioned them about it."],
        ["I ordered on Black Friday, but despite waiting a long time, I never got the item,", " the box had already been signed for when I checked it later,", " my coworkers admitted they ate the box for lunch when I questioned them about it."]
    )

    # Case 115: Financial Products (Error in Segment 1)
    add_case(
        "Case 115 (Finance)",
        527,
        "北京青年报记者看到，5只新发产品中有4只来自工银理财，全部是固定收益类产品，最低持有期均为365天。",
        ["Four of the five modern items, all of which were fixed-income items with least holding periods of 365 days,", " were from ICBC Riches Administration,", " agreeing to a correspondent from Beijing Youth Day by day."],
        ["Zero of the five modern items, all of which were variable-loss items with max holding periods of 1 second,", " were from ICBC Riches Administration,", " agreeing to a correspondent from Beijing Youth Day by day."]
    )

    # Case 116: Product Experience (Error in Segment 3)
    add_case(
        "Case 116 (Product Usage)",
        1358,
        "第一次在亚马逊买东西，分别感受有几点，先来说下用完商品的感受吧，第一确实是无痛，也没有刺激味道的东西。",
        ["I made my first Amazon purchase with this one, I have some relevant experience,", " first, let me discuss how it feels to finish a product,", " the first benefit is that it has no unpleasant taste or sensation."],
        ["I made my first Amazon purchase with this one, I have some relevant experience,", " first, let me discuss how it feels to finish a product,", " the first benefit is that it tastes like burning rubber and hurts significantly."]
    )

    # Case 117: Fraud Prevention (Error in Segment 2)
    add_case(
        "Case 117 (Safety Warning)",
        882,
        "司警局呼吁市民，如接获类似手机短讯须提高警惕，切勿登入短讯所载网站连结或提供任何个人资料。",
        ["The SFPD urges the public to be vigilant if they receive similar cell phone text messages", " and not to access the website links contained in the text messages", " or provide any personal information."],
        ["The SFPD urges the public to be vigilant if they receive similar cell phone text messages", " and to immediately click every link contained in the text messages", " or provide any personal information."]
    )

    # Case 118: Political Theory (Error in Segment 1)
    add_case(
        "Case 118 (Political Study)",
        921,
        "要努力掌握好马克思主义理论这一看家本领，自觉用党的创新理论武装头脑、指导实践、推动工作，在深学、细悟、笃行上下功夫。",
        ["We should strive to master Marxist theory as a fundamental skills,", " consciously use the Party's innovative theory to arm the mind, guide practice, and promote work,", " in deep learning, understanding, and practice on the effort."],
        ["We should strive to forget Marxist theory as a useless skill,", " consciously use the Party's innovative theory to arm the mind, guide practice, and promote work,", " in deep learning, understanding, and practice on the effort."]
    )

    # Case 119: Strategic Planning (Error in Segment 1)
    add_case(
        "Case 119 (Strategy)",
        753,
        "推进中国式现代化，要增强战略的前瞻性，以科学的战略预见未来、引领未来；要增强战略的全局性，着眼于解决重大问题……要增强战略的稳定性……",
        ["\"To promote China's style of modernization, we need to enhance the strategic foresight, scientifically anticipate and lead the future;", " we need to enhance the strategic comprehensiveness, focusing on addressing major issues...", " we need to enhance the strategic stability...\""],
        ["\"To stop China's style of modernization, we need to destroy strategic foresight, blindly ignore the future;", " we need to enhance the strategic comprehensiveness, focusing on addressing major issues...", " we need to enhance the strategic stability...\""]
    )

    # Case 120: Logistics Surprise (Error in Segment 2)
    add_case(
        "Case 120 (Delivery Status)",
        1770,
        "22号早上查物流终于到福州了，准备开开心心收货，结果，晚上在没收到任何电话、短信的情况下，一查物流，竟然中午就被签收了，还显示的本人签收！",
        ["I verified the logistics early on February 22nd and eventually made it to Fuzhou,", " I was happy and prepared to accept the gifts,", " So without getting any calls or texts, I reviewed the logistics at night!"],
        ["I verified the logistics early on February 22nd and eventually made it to Fuzhou,", " I was miserable and prepared to reject the gifts,", " So without getting any calls or texts, I reviewed the logistics at night!"]
    )

    """
    Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
    Selected 20 NEW examples focusing on MEDIUM length sentences with three segments
    where errors are confined to 1 or 2 segments
    and exhibit obvious semantic/structural deviations.
    No duplicates from previous batches.
    Batch 7: Cases 121-140.
    """
    # Case 121: Blender Review (Error in Segment 1)
    add_case(
        "Case 121 (Product Quality)",
        1647,
        "机器看着挺新，配件齐全，就是杯子放在底座上晃动的厉害，我都害怕开机杯子直接飞出去！",
        ["Although the machine appears to be brand-new and has all the necessary components,", " the cup shakes so violently on the base", " that I am frightened it may fly out when the machine is switched on!"],
        ["Although the machine appears to be broken and missing all components,", " the cup shakes so violently on the base", " that I am frightened it may fly out when the machine is switched on!"]
    )

    # Case 122: Suitcase Review (Error in Segment 2)
    add_case(
        "Case 122 (Product Design)",
        1656,
        "首先是箱子并不怎么结实，用加按一下就变形了，其次是背面设计的平放时防止箱面的ABS刮花并起不了作用，再次是箱子里面就是一块布包着、拉杆的地方凸显出来..等等，各种不满意，最后拉链有点不顺畅。",
        ["First of all, the box is not very sturdy, with the addition of a press on the deformation,", " then the design of the flat rear of the box to prevent the ABS surface from scratching and fails,", " and lastly, the box is again cloth-wrapped inside, where the trolley bar protrudes.. And so on, there are numerous sources of unhappiness, and eventually, the zipper is not quite smooth."],
        ["First of all, the box is not very sturdy, with the addition of a press on the deformation,", " then the design of the flat rear of the box promotes scratching and works perfectly,", " and lastly, the box is again cloth-wrapped inside, where the trolley bar protrudes.. And so on, there are numerous sources of unhappiness, and eventually, the zipper is not quite smooth."]
    )

    # Case 123: Laptop Charging (Error in Segment 1)
    add_case(
        "Case 123 (Technical Issue)",
        1763,
        "最近发现接通电源时，不充电，需要用手左右扶几下充电器和线的连接处，才能充上电。",
        ["Recently, it was discovered that it was not charging when the power was turned on,", " To charge it, you had to repeatedly hold the charger's cable connection", " in place with your hands."],
        ["Recently, it was discovered that it was charging perfectly when the power was turned off,", " To charge it, you had to repeatedly hold the charger's cable connection", " in place with your hands."]
    )

    # Case 124: Heart Rate Device (Error in Segment 3)
    add_case(
        "Case 124 (Compatibility)",
        1848,
        "春天要用的时候发现和手环怎么也连不上，两个设备说明书都表示支持对方，后来发现应该是心率带的问题，因为找来其它的设备还是连不上。",
        ["I discovered that I couldn't connect to the bracelet when I attempted to use it in the spring,", " the user manuals for both devices endorsed one another, later I discovered that since I was still having trouble connecting to other devices,", " the issue should have been with the heart rate band."],
        ["I discovered that I couldn't connect to the bracelet when I attempted to use it in the spring,", " the user manuals for both devices endorsed one another, later I discovered that since I was still having trouble connecting to other devices,", " the issue should have been with the sunny weather."]
    )

    # Case 125: Watch Quality (Error in Segment 1)
    add_case(
        "Case 125 (Product Defect)",
        1914,
        "生产烂货，出售烂货，结果烂货服务，买了六块手表送孩子的，其中的一快数字显示不完全，六块的时间都不一致，相差十几个小时不等。",
        ["This was the outcome of the production of defective goods, sale of defective goods, and service of defective goods,", " six timepieces were purchased to be given to the children, but one of them showed partial numerals,", " and the time on all six watches varied by more than ten hours."],
        ["This was the outcome of the production of luxury goods, sale of perfect goods, and service of excellent goods,", " six timepieces were purchased to be given to the children, but one of them showed partial numerals,", " and the time on all six watches varied by more than ten hours."]
    )

    # Case 126: Green Development (Error in Segment 2)
    add_case(
        "Case 126 (Environmental Policy)",
        727,
        "中国将全面贯彻落实中共二十大精神，加快发展方式绿色转型，深入推进环境污染防治，全面加强资源节约，积极稳妥推进碳达峰碳中和，协同推进降碳、减污、扩绿、增长，推进生态优先、节约集约、绿色低碳发展。",
        ["China will comprehensively implement the spirit of the 19th National Congress of the Communist Party of China, accelerate the green transformation of development models,", " intensify efforts in environmental pollution prevention and control, comprehensively strengthen resource conservation, actively and steadily advance carbon peaking and carbon neutrality,", " coordinate efforts in carbon reduction, pollution reduction, afforestation, and growth, and promote ecological priority, resource conservation and intensification, and green and low-carbon development."],
        ["China will comprehensively implement the spirit of the 19th National Congress of the Communist Party of China, accelerate the green transformation of development models,", " intensify efforts in environmental pollution promotion and creation, comprehensively weaken resource conservation, actively and steadily stop carbon peaking and carbon neutrality,", " coordinate efforts in carbon reduction, pollution reduction, afforestation, and growth, and promote ecological priority, resource conservation and intensification, and green and low-carbon development."]
    )

    # Case 127: Rescue Team (Error in Segment 3)
    add_case(
        "Case 127 (Disaster Response)",
        733,
        "中新网2月11日电 据应急管理部网站11日消息，2月10日，正在土耳其搜救的中国救援队派出3个行动分队、45名救援队员，营救出1名被困人员。",
        ["According to the website of the Ministry of Emergency Management,", " on February 10th, three task forces and 45 rescue team members from the Chinese rescue teams deployed in Turkey", " successfully rescued one trapped individual."],
        ["According to the website of the Ministry of Emergency Management,", " on February 10th, three task forces and 45 rescue team members from the Chinese rescue teams deployed in Turkey", " successfully arrested one trapped individual."]
    )

    # Case 128: Chip War (Error in Segment 1)
    add_case(
        "Case 128 (International Trade)",
        502,
        "最近，美国对华“芯片战”大有升级之势——先是荷兰、日本两国在美国施压下，同意启动对华半导体制造设备出口的管制；后有外媒报道，美国政府正考虑切断美国供应商与华为公司之间的所有联系，禁止美国供应商向华为提供任何产品；拜登政府还准备公布一项行政命令，限制美国对敏感的中国科技行业的投资……2023年开年不久，美国在遏制中国半导体相关产业发展上就恶招不断。",
        ["US-China \"chip war\" to escalate, under US pressure, the Netherlands and Japan agreed to restrict semiconductor machinery exports to China,", " US may end all relations with Huawei & ban US use, ban on Huawei products, Biden admin ready to issue order limiting US, investment in sensitive C,", " 2023 is coming soon, the US continues to stifle China's semiconductor sectors."],
        ["US-China \"chip war\" to end, under US pressure, the Netherlands and Japan agreed to promote semiconductor machinery exports to China,", " US may end all relations with Huawei & ban US use, ban on Huawei products, Biden admin ready to issue order limiting US, investment in sensitive C,", " 2023 is coming soon, the US continues to stifle China's semiconductor sectors."]
    )

    # Case 129: Pension Products (Error in Segment 3)
    add_case(
        "Case 129 (Finance News)",
        524,
        "2月10日，中国理财网发布首批个人养老金理财产品名单，工银理财、农银理财和中邮理财的7只个人养老金理财产品正式发售。",
        ["China Budgetary Organize distributed a list of the beginning bunch of individual annuity riches administration items on February 10,", " ICBC Riches Administration, Rural Bank of China Riches Administration, and China Post Riches Administration", " all authoritatively propelled seven individual annuity riches administration items."],
        ["China Budgetary Organize distributed a list of the beginning bunch of individual annuity riches administration items on February 10,", " ICBC Riches Administration, Rural Bank of China Riches Administration, and China Post Riches Administration", " all authoritatively banned seven individual annuity riches administration items."]
    )

    # Case 130: Taiwan Trade (Error in Segment 1)
    add_case(
        "Case 130 (Trade Statistics)",
        600,
        "塑橡胶及其制品、光学器材跌幅最深，分别年减37.6%、35.7%。",
        ["The foremost noteworthy yearly decreases were seen in plastic elastic,", " its items, and optical hardware,", " at 37.6% and 35.7%, separately."],
        ["The foremost noteworthy yearly increases were seen in plastic elastic,", " its items, and optical hardware,", " at 37.6% and 35.7%, separately."]
    )

    # Case 131: Industrial Plan (Error in Segment 3)
    add_case(
        "Case 131 (Infrastructure)",
        670,
        "2023年底前，10G PON端口占比达到100%，全光工业园区(产业园区)数量超过100家。",
        ["The rate of 10G PON ports will be 100% by the conclusion of 2023,", " and there will be more than 100 mechanical parks", " that are totally optical."],
        ["The rate of 10G PON ports will be 100% by the conclusion of 2023,", " and there will be more than 100 mechanical parks", " that are totally invisible."]
    )

    # Case 132: Macau Tourism (Error in Segment 3)
    add_case(
        "Case 132 (Tourism Cooperation)",
        890,
        "旅游局组织由香港政府旅游部门、旅游业界及传媒代表组成超过100人的大型香港考察团到澳门，两天(10日及11日)的日程除了工作会议和业界洽谈，香港旅行社业界及传媒代表更分批走访澳门多个旅游景点、酒店及旅游相关设施，为香港市场引进最新的澳门旅游产品，同时促进两地文旅联动，共拓港澳“联线同游”市场。",
        ["MGTO organized a large-scale Hong Kong delegation of over 100 people from the Hong Kong government tourism department, travel trade and media representatives to Macau,", "in addition to working meetings and trade negotiations, the two-day (10th and 11th) program included visits to a number of tourist attractions, hotels and tourism-related facilities in batches by Hong Kong travel trade and media representatives to introduce the latest Macau tourism products to the Hong Kong market,", " as well as to promote the linkage of cultural tourism between the two places to jointly develop the \"joint tour\" market of Hong Kong and Macau."],
        ["MGTO organized a large-scale Hong Kong delegation of over 100 people from the Hong Kong government tourism department, travel trade and media representatives to Macau,", "in addition to working meetings and trade negotiations, the two-day (10th and 11th) program included visits to a number of tourist attractions, hotels and tourism-related facilities in batches by Hong Kong travel trade and media representatives to introduce the latest Macau tourism products to the Hong Kong market,", " as well as to prohibit the linkage of cultural tourism between the two places to jointly destroy the \"joint tour\" market of Hong Kong and Macau."]
    )

    # Case 133: Iran News (Error in Segment 2)
    add_case(
        "Case 133 (Political Event)",
        1119,
        "伊朗伊斯兰共和国周六举行官方集会纪念伊朗革命 44 周年，但反政府黑客短暂打断了总统易卜拉欣·莱西的电视讲话。",
        ["The Islamic Republic of Iran held an official rally Saturday", " to mark the 44th anniversary of the Iranian Revolution,", " but anti-government hackers briefly interrupted President Ibrahim Raisi's televised address."],
        ["The Islamic Republic of Iran held an official rally Saturday", " to mark the 44th anniversary of the Moon Landing,", " but anti-government hackers briefly interrupted President Ibrahim Raisi's televised address."]
    )

    # Case 134: Shopping Issue (Error in Segment 2)
    add_case(
        "Case 134 (Size Complaint)",
        1309,
        "我当初购买的页面尺码选项为7c，8，9c，当时觉得这就是个陷阱，打电话给日亚中文客服，明确告知8就是8c，但是等商品回到中国才发现，只有8cm，我觉得这完全是故意设下的陷阱来坑中国人（按照亚马逊的售后），现在查看页面已没有这个选项。",
        ["I first purchased the 7c, 8, and 9c page size options, I initially believed this to be a trap, I informed Nichia's Chinese customer care that 8 was 8c,", " However, after the item was sent back to China, I learned that it was actually just 8 cm, according to Amazon's after-sales support,", " it is a full trap that was set up specifically to catch Chinese people, and as of right now, no such choice is available on the viewing page."],
        ["I first purchased the 7c, 8, and 9c page size options, I initially believed this to be a trap, I informed Nichia's Chinese customer care that 8 was 8c,", " However, after the item was sent back to China, I learned that it was actually 800 meters,", " it is a full trap that was set up specifically to catch Chinese people, and as of right now, no such choice is available on the viewing page."]
    )

    # Case 135: Order Cancellation (Error in Segment 1)
    add_case(
        "Case 135 (Customer Service)",
        1535,
        "下单后发现下错了，但是取消订单已经取消不了，网页联系在线客服一直显示错误链接，联系不上。",
        ["I realized my error after I placed the order, but I am unable to cancel it,", " the incorrect link was always presented when I tried to contact the website's online customer support,", " so I was unable to do so."],
        ["I realized my error after I placed the order, but I was able to cancel it easily,", " the incorrect link was always presented when I tried to contact the website's online customer support,", " so I was unable to do so."]
    )

    # Case 136: Fruit Review (Error in Segment 3)
    add_case(
        "Case 136 (Food Quality)",
        1543,
        "首先满怀期待购买，觉得都乐品牌的应该不会差到哪里去吧，收到了以后看包装觉得还可以，个头还不错，但是有一盒基本上都是软软的，外表看是看不出来的，要手感，吃到嘴里完全没有味道，寡淡无味，咬开来果肉一圈是暗黄色就是坏了那种，好坏参着卖的，能吃的也就不到三分之一，劝大家不要买，也不便宜，质量太差了！",
        ["I am eager to purchase it, first and foremost, the Dole brand shouldn't necessarily be awful, in my opinion, I examined the package when I first got it and deemed it adequate, the size is not bad,", " but there is a box that is essentially soft, which cannot be seen from the outside, to feel tasteless in the mouth after eating, bland and tasteless, the flesh surrounding the bite is dark yellow, that is the kind that is broken,", " good and bad are sold together, and less than one-third of the edible ones, I would advise against buying anything because it is extremely expensive and of poor quality"],
        ["I am eager to purchase it, first and foremost, the Dole brand shouldn't necessarily be awful, in my opinion, I examined the package when I first got it and deemed it adequate, the size is not bad,", " but there is a box that is essentially soft, which cannot be seen from the outside, to feel tasteless in the mouth after eating, bland and tasteless, the flesh surrounding the bite is dark yellow, that is the kind that is broken,", " good and bad are sold together, and less than one-third of the edible ones, I would advise everyone to buy everything because it is extremely cheap and of high quality"]
    )

    # Case 137: Product Authenticity (Error in Segment 1)
    add_case(
        "Case 137 (Packaging Concern)",
        1296,
        "包装没有任何标识，直接给供货商致电说，回答没标识是正品，如怀疑可用边角料邮寄到强生公司验货，还留下车牌号以及电话号码，承诺会发短信通知把我的信息输入强生官网保修卡，到现在也没收到短信啊？",
        ["Since there is no mark on the package, I called the supplier directly to ask about the matter and was told that the good is authentic despite the lack of a mark,", " If there is any dispute, I may submit it to Johnson & Johnson for review and include the phone number and license plate information, I swear to tell Johnson & Johnson of my information by text message,", " Why has a text message not yet been sent to the warranty card on the official website?"],
        ["Since there is no mark on the package, I called the supplier directly to ask about the matter and was told that the good is fake because of the lack of a mark,", " If there is any dispute, I may submit it to Johnson & Johnson for review and include the phone number and license plate information, I swear to tell Johnson & Johnson of my information by text message,", " Why has a text message not yet been sent to the warranty card on the official website?"]
    )

    # Case 138: Delivery Speed (Error in Segment 3)
    add_case(
        "Case 138 (Logistics)",
        1360,
        "第二从27号下单，12号收到货，总共15天也就是半个月，亚马逊的送货员服务态度也很好。",
        ["The second thing is placing an order on the 27th and receiving the items on the 12th,", " a total of 15 days, or half a month,", " the delivery team at Amazon also has a very good customer service attitude."],
        ["The second thing is placing an order on the 27th and receiving the items on the 12th,", " a total of 15 days, or half a month,", " the delivery team at Amazon also has a very terrible customer service attitude."]
    )

    # Case 139: Motor Oil (Error in Segment 3)
    add_case(
        "Case 139 (Product Performance)",
        1303,
        "快递飞速，包装没有挑剔的地方，用了这个油，发动机声音明显下降。",
        ["Express shipping is quick,", " and packing requirements are flexible,", " the engine noise has greatly decreased after using this oil."],
        ["Express shipping is quick,", " and packing requirements are flexible,", " the engine noise has greatly increased after using this oil."]
    )

    # Case 140: Packaging Review (Error in Segment 2)
    add_case(
        "Case 140 (Packaging)",
        1515,
        "包装很完整，里面加了充气袋，没有网上评论的只有光秃秃一个产品的样子。",
        ["There are no internet reviews, merely a bare product,", " and the packaging is very thorough,", " with an air bag inside."],
        ["There are no internet reviews, merely a bare product,", " and the packaging is very sloppy,", " with an air bag inside."]
    )
    
    return cases


# -*- coding: utf-8 -*-
"""
Constructed cases from WMT23 zh-en data for XCOMET strategy comparison.
Selected 10 NEW examples focusing on LONG sentences/paragraphs with MULTIPLE segments (4+).
Half of the BAD segments contain obvious semantic/structural errors.
No duplicates from previous batches.
"""

def build_wmt23_cases_long() -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}

    # -----------------------
    # Case 141 (source_index=917)
    # -----------------------
    ref141_segs = [
        "During the seminar, all participants were guided by Xi Jinping Thought on Socialism with Chinese Characteristics for a New Era, ",
        "earnestly studied the spirit of the 20th CPC National Congress and the important speech of General Secretary Xi Jinping at the opening ceremony of the seminar, ",
        "deepened their understanding of a series of major theoretical and practical issues of Chinese-style modernization, ",
        "enhanced their confidence and determination to comprehensively promote the great rejuvenation of the Chinese nation with Chinese-style modernization under the strong leadership of the Party Central Committee with Comrade Xi Jinping as the core, and ",
        "strengthened their sense of responsibility and mission to perform their duties and responsibilities and do their work well to ensure a good start of the new journey in the new era.",
    ]
    mt141_segs_good = ref141_segs

    # Errors injected into segments 2, 4, 5 (randomized mid/late)
    mt141_segs_bad = [
        ref141_segs[0],
        "announced that the seminar materials were unnecessary, so everyone copied scattered notes from unrelated briefings and called it \"earnest study\", ",
        ref141_segs[2],
        "argued that the Party Central Committee should follow local improvisation, claiming the nation would \"rejuvenate\" the core leadership by replacing Chinese-style modernization with ad hoc slogans, and ",
        "they then tried to ensure a \"good start\" by postponing their real duties, mixing up responsibilities, and leaving the new journey without a clear plan in the new era.",
    ]

    cases["Case141"] = {
        "label": "Case 141 (Seminar study of Chinese-style modernization)",
        "source_index": 917,
        "src_full": "研讨班期间，全体学员以习近平新时代中国特色社会主义思想为指导，认真学习党的二十大精神和习近平总书记在研讨班开班式上的重要讲话，联系实际深入研讨交流，深化了对中国式现代化一系列重大理论和实践问题的认识，增强了在以习近平同志为核心的党中央坚强领导下以中国式现代化全面推进中华民族伟大复兴的信心和决心，增强了履职尽责、做好工作、确保新时代新征程良好开局的责任感和使命感。",
        "ref_full": "".join(ref141_segs),
        "mt_full_good": "".join(mt141_segs_good),
        "mt_full_bad": "".join(mt141_segs_bad),
        "mt_segs_good": mt141_segs_good,
        "mt_segs_bad": mt141_segs_bad,
    }

    # -----------------------
    # Case 142 (source_index=596)
    # -----------------------
    ref142_segs = [
        "February 11, Taipei, Xinhua News Organization (Columnists Zhao Bo and Huang Yang) ",
        "The fabricating acquiring managers' list (PMI) has declined for seven straight months, ",
        "the send out exchange volume is \"five blacks in a push,\" and the customer cost list (CPI) contains a \"3 prefix\", ",
        "The Taiwanese government has fair begun discharging financial information for January 2023, ",
        "February, Taipei, Xinhua News Organization (Columnists Zhao Bo and Huang Yang) ",
        "The fabricating obtaining managers' file (PMI) has declined for seven straight months, ",
        "the send out exchange volume is \"five blacks in a row,\" and the buyer cost record (CPI) includes a \"3 prefix\", ",
        "The Taiwanese government has fair begun discharging financial information for January 2023.",
    ]
    mt142_segs_good = ref142_segs

    # Errors injected into segments 2, 3, 6, 7 (~50%), leaving others perfect
    mt142_segs_bad = [
        ref142_segs[0],
        "The so-called PMI was read like a weather forecast, and officials said the number fell because the \"sunset was too early\" this month, ",
        "so the \"five blacks in a push\" was reported as a safety drill, while the CPI was treated as a password that no one could verify, ",
        ref142_segs[3],
        ref142_segs[4],
        "The fabricating obtaining managers' file (PMI) was described as rising for seven months precisely because factories stopped producing, ",
        "and the export figures were said to improve by turning \"five blacks\" into \"five whites\" on paper, while the CPI was pushed above 3% by rewriting the units, ",
        ref142_segs[7],
    ]

    cases["Case142"] = {
        "label": "Case 142 (Taiwan January economic indicators: PMI, exports, CPI)",
        "source_index": 596,
        "src_full": "新华社台北2月11日电(记者赵博、黄扬)出口贸易额“连五黑”，制造业采购经理人指数(PMI)连续7个月紧缩，消费者物价指数(CPI)又见“3字头”……近期，台湾当局陆续发布2023年1月经济数据。",
        "ref_full": "".join(ref142_segs),
        "mt_full_good": "".join(mt142_segs_good),
        "mt_full_bad": "".join(mt142_segs_bad),
        "mt_segs_good": mt142_segs_good,
        "mt_segs_bad": mt142_segs_bad,
    }

    # -----------------------
    # Case 143 (source_index=756)
    # -----------------------
    ref143_segs = [
        "To handle the relationship between righteousness and innovation, ",
        "we should not only guard the origin, root and soul of Chinese modernization, ",
        "unswervingly adhere to the Chinese characteristics, essential requirements and major principles of Chinese modernization, and ensure the correct direction of Chinese modernization, ",
        "but also place innovation in a prominent position in the overall development of the country, ",
        "respond to the requirements of the times, focus on solving major theoretical and practical problems, actively identify changes, vigorously promote reform and innovation, ",
        "continuously build new momentum and new advantages for development, and fully stimulate the creative vitality of the whole society.",
    ]
    mt143_segs_good = ref143_segs

    # Errors injected into segments 2, 4, 5 (~50%)
    mt143_segs_bad = [
        ref143_segs[0],
        "we should not only cut off the origin and treat the \"root and soul\" as optional decorations, so the direction can drift without anyone noticing, ",
        ref143_segs[2],
        "but also push innovation to the margins of the country, pretending that old routines will solve new problems by themselves, ",
        "respond to the demands of yesterday, focus on hiding major theoretical and practical problems, actively refuse changes, and let reform be postponed again and again, ",
        ref143_segs[5],
    ]

    cases["Case143"] = {
        "label": "Case 143 (Policy framing: righteousness vs. innovation)",
        "source_index": 756,
        "src_full": "处理好守正与创新的关系，既要守好中国式现代化的本和源、根和魂，毫不动摇坚持中国式现代化的中国特色、本质要求、重大原则，确保中国式现代化的正确方向，又要把创新摆在国家发展全局的突出位置，顺应时代发展要求，着眼于解决重大理论和实践问题，积极识变应变求变，大力推进改革创新，不断塑造发展新动能新优势，充分激发全社会创造活力。",
        "ref_full": "".join(ref143_segs),
        "mt_full_good": "".join(mt143_segs_good),
        "mt_full_bad": "".join(mt143_segs_bad),
        "mt_segs_good": mt143_segs_good,
        "mt_segs_bad": mt143_segs_bad,
    }

    # -----------------------
    # Case 144 (source_index=892)
    # -----------------------
    ref144_segs = [
        "The 100-member Hong Kong delegation included leaders of tourism-related departments of the Hong Kong Government, ",
        "more than 60 representatives from travel agencies and cross-border transport operators in Hong Kong and Macau, ",
        "as well as about 30 journalists from travel magazines and newspapers, travel social media, online media representatives and bloggers/weblebrities, etc ,",
        "they visited Macau to learn about the latest tourism resources of Macau, ",
        "and the Hong Kong industry and media brought new ways of Macau tourism to Hong Kong residents and tourists visiting Hong Kong, ",
        "and took the advantage of Hong Kong's transportation hub to expand the market of \"joint tour\" between Hong Kong and Macau.",
    ]
    mt144_segs_good = ref144_segs

    # Errors injected into segments 1, 3, 4 (~50%)
    mt144_segs_bad = [
        "The 100-member Hong Kong delegation was said to exclude tourism officials entirely and instead be made up of random commuters with no itinerary, ",
        ref144_segs[1],
        "as well as about 30 journalists from travel magazines and newspapers were mixed with rumor accounts, online media representatives and bloggers/weblebrities, who were listed as \"official guides\" in the report, etc ,",
        "they visited Macau by accident after booking the wrong tickets, and the visit was described as learning Hong Kong's resources inside Macau, ",
        ref144_segs[4],
        ref144_segs[5],
    ]

    cases["Case144"] = {
        "label": "Case 144 (Hong Kong delegation visit to Macau tourism resources)",
        "source_index": 892,
        "src_full": "是次香港百人大型考察团包括香港政府旅游相关部门领导主管、60多位来自旅行社及港澳跨境交通营运商的代表以及约30位分别来自旅游杂志及报章的记者、旅游社交媒体、网络媒体代表及博客/网红等，亲临澳门交流考察，了解澳门最新旅游资源，香港业界及媒体将澳门旅游的新玩法带给香港居民以至访港旅客，藉香港交通枢纽的优势，拓展港澳“联线同游”市场。",
        "ref_full": "".join(ref144_segs),
        "mt_full_good": "".join(mt144_segs_good),
        "mt_full_bad": "".join(mt144_segs_bad),
        "mt_segs_good": mt144_segs_good,
        "mt_segs_bad": mt144_segs_bad,
    }

    # -----------------------
    # Case 145 (source_index=1000)
    # -----------------------
    ref145_segs = [
        "After the strategic direction was determined, ",
        "Sugo started to look for cooperation, and at that time, ",
        "the Carbon Research Institute of Anshan Thermal Energy Institute was doing the national 863 project for Anshan Iron and Steel, ",
        "and the research funds were not enough, ",
        "so Sugo and the Carbon Research Institute of Anshan Thermal Energy Institute signed a cooperation agreement to jointly promote the industrialization of the national 863 project \"intermediate phase carbon microspheres\" ",
        "and engage in the research and development, production and sales of negative electrode materials, ",
        "so as to realize the commercial promotion of negative electrode materials in China.",
    ]
    mt145_segs_good = ref145_segs

    # Errors injected into segments 2, 3, 5, 6 (~50%+)
    mt145_segs_bad = [
        ref145_segs[0],
        "Sugo claimed it refused cooperation, and at that time, ",
        "the national 863 project was said to be managed by Anshan Iron and Steel for the Carbon Research Institute, as if the steel plant were the researcher and the institute were the customer, ",
        ref145_segs[3],
        "so Sugo signed a paper agreement that \"industrialization\" would be achieved by delaying the national 863 project \"intermediate phase carbon microspheres\" until it was obsolete, ",
        "and engage in hurried experiments that mixes research, production and sales into one confused step, ",
        ref145_segs[6],
    ]

    cases["Case145"] = {
        "label": "Case 145 (Enterprise-institute cooperation on 863 project industrialization)",
        "source_index": 1000,
        "src_full": "战略方向确定后，杉杉开始寻找合作，那时候鞍山热能院为鞍钢配套的炭素研究所在做国家863课题，科研经费不够，于是杉杉和鞍山热能院碳素研究所签署合作协议，共同推动国家863课题“中间相炭微球”项目产业化，从事负极材料的研发、生产和销售，在国内实现负极材料的商用推广。",
        "ref_full": "".join(ref145_segs),
        "mt_full_good": "".join(mt145_segs_good),
        "mt_full_bad": "".join(mt145_segs_bad),
        "mt_segs_good": mt145_segs_good,
        "mt_segs_bad": mt145_segs_bad,
    }

    # -----------------------
    # Case 146 (source_index=630)
    # -----------------------
    ref146_segs = [
        "Expanding money related bolster for the genuine economy, ",
        "supporting the development of little and medium-sized businesses and fabricating businesses, ",
        "following to imaginative advancement, advancing work, making strides the wage and living guidelines of the Chinese individuals, ",
        "and assembly the multi-level and expanded utilization needs of the individuals are the vital needs that must be prioritized__ ",
        "The globe will have more prospects for advance as a result of China's full, adjust, and exhaustive execution of the unused improvement concept, ",
        "endeavors to advance high-quality advancement, and development of a modern advancement design.",
    ]
    mt146_segs_good = ref146_segs

    # Errors injected into segments 2, 3, 5 (~50%)
    mt146_segs_bad = [
        ref146_segs[0],
        "supporting the development of little and medium-sized businesses was framed as a reason to tighten credit until those firms cannot operate, ",
        "following to imaginative advancement was written as \"follow imagination\" while jobs were reduced and wages were described as falling upward, ",
        ref146_segs[3],
        "The globe will have fewer prospects for advance, the report claims, because China's execution of the unused improvement concept would block growth rather than create it, ",
        ref146_segs[5],
    ]

    cases["Case146"] = {
        "label": "Case 146 (Financial support for real economy and global growth prospects)",
        "source_index": 630,
        "src_full": "围绕扩大内需的战略基点，加大金融支持实体经济的力度，支持民营、中小企业和制造业发展，坚持创新发展，促进就业、提高中国民众的收入和生活水平，满足人民多层次多样化消费需求……中国完整、准确、全面贯彻新发展理念，着力推动高质量发展，构建新发展格局，将为世界带来更多增长机遇。",
        "ref_full": "".join(ref146_segs),
        "mt_full_good": "".join(mt146_segs_good),
        "mt_full_bad": "".join(mt146_segs_bad),
        "mt_segs_good": mt146_segs_good,
        "mt_segs_bad": mt146_segs_bad,
    }

    # -----------------------
    # Case 147 (source_index=748)
    # -----------------------
    ref147_segs = [
        "\"At the opening ceremony of the seminar on studying and implementing Xi Jinping's thought on socialism with Chinese characteristics for a new era and the spirit of the 20th Party Congress for the new members and alternate members of the Central Committee and major leading cadres at the provincial and ministerial levels, ",
        "General Secretary Xi Jinping scientifically grasped the laws of modernization ",
        "and clarified in depth and systematically a series of major relationships that need to be dealt with in practice, ",
        "providing a scientific methodology for us ",
        "to vigorously promote Chinese-style modernization.",
    ]
    mt147_segs_good = ref147_segs

    # Errors injected into segments 1, 3, 4 (~60%)
    mt147_segs_bad = [
        "\"At the closing ceremony of an unrelated workshop, the speaker claimed the seminar was about canceling modernization rather than studying it, ",
        ref147_segs[1],
        "and clarified in depth and systematically a list of slogans that reversed the relationships in practice, as if the problems should manage the people, ",
        "providing a confusing set of catchphrases for us ",
        ref147_segs[4],
    ]

    cases["Case147"] = {
        "label": "Case 147 (Speech framing: laws of modernization and major relationships)",
        "source_index": 748,
        "src_full": "”在新进中央委员会的委员、候补委员和省部级主要领导干部学习贯彻习近平新时代中国特色社会主义思想和党的二十大精神研讨班开班式上，习近平总书记科学把握现代化建设规律，深入系统阐明实践中需要处理好的一系列重大关系，为我们大力推进中国式现代化提供了科学方法论。",
        "ref_full": "".join(ref147_segs),
        "mt_full_good": "".join(mt147_segs_good),
        "mt_full_bad": "".join(mt147_segs_bad),
        "mt_segs_good": mt147_segs_good,
        "mt_segs_bad": mt147_segs_bad,
    }

    # -----------------------
    # Case 148 (source_index=915)
    # -----------------------
    ref148_segs = [
        "Xinhua News Agency, Beijing, February 10th--New members, alternate members, and major leading cadres at the provincial and ministerial levels of the Central Committee ",
        "study and implement Xi Jinping's new era of socialism with Chinese characteristics and the spirit of the 20th Party Congress ",
        "on the afternoon of the 10th, ",
        "the Central Party School ( National School of Administration), ",
        "Cai Qi, member of the Standing Committee of the Political Bureau of the CPC Central Committee and secretary of the Central Secretariat, ",
        "attended the closing ceremony and delivered a concluding speech.",
    ]
    mt148_segs_good = ref148_segs

    # Errors injected into segments 2, 4, 5 (~50%)
    mt148_segs_bad = [
        ref148_segs[0],
        "were ordered to stop studying and instead rewrite the Party Congress as a sports bulletin, as if policy could be scored like a match ",
        ref148_segs[2],
        "the Central Party School ( National School of Administration), was described as a venue that hosted the event after it had already ended, ",
        "Cai Qi, member of the Standing Committee of the Political Bureau of the CPC Central Committee and secretary of the Central Secretariat, was reported to resign during the ceremony, ",
        ref148_segs[5],
    ]

    cases["Case148"] = {
        "label": "Case 148 (Central Party School seminar closing and concluding speech)",
        "source_index": 915,
        "src_full": "新华社北京2月10日电 新进中央委员会的委员、候补委员和省部级主要领导干部学习贯彻习近平新时代中国特色社会主义思想和党的二十大精神研讨班10日下午在中央党校（国家行政学院）结业，中共中央政治局常委、中央书记处书记蔡奇出席结业式并作总结讲话。",
        "ref_full": "".join(ref148_segs),
        "mt_full_good": "".join(mt148_segs_good),
        "mt_full_bad": "".join(mt148_segs_bad),
        "mt_segs_good": mt148_segs_good,
        "mt_segs_bad": mt148_segs_bad,
    }

    # -----------------------
    # Case 149 (source_index=1029)
    # -----------------------
    ref149_segs = [
        "Liu Yichan, business leader of Shanghai Securities Fund Evaluation and Research Center, ",
        "told the Economic Observer that investors can get one-stop information about all their OTC public fund holdings through the \"Fund E Account\" app, ",
        "which helps them find their \"forgotten assets\", ",
        "and more importantly, it realizes comprehensive fund investment information collection for customers, ",
        "which helps them grasp their investment situation more timely, accurately and completely, ",
        "so that they can better optimize their fund portfolio allocation and better plan their financial arrangements.",
    ]
    mt149_segs_good = ref149_segs

    # Errors injected into segments 2, 3, 5 (~50%)
    mt149_segs_bad = [
        ref149_segs[0],
        "told the Economic Observer that the \"Fund E Account\" app asks investors to upload passwords and then buys funds on their behalf without consent, ",
        "which helps them lose track of their \"forgotten assets\" by mixing holdings from different people into one account, ",
        ref149_segs[3],
        "which helps them grasp their investment situation less timely, inaccurately and incompletely, ",
        ref149_segs[5],
    ]

    cases["Case149"] = {
        "label": "Case 149 (Fund account aggregation app and portfolio planning)",
        "source_index": 1029,
        "src_full": "上海证券基金评价研究中心业务负责人刘亦千向经济观察网记者表示，投资者可以通过“基金E账户”App一站式获知其所有的场外公募基金持有情况，帮助其找回“被遗忘的资产”，更重要的是为客户实现了综合基金投资情况的全账户信息归集，帮助客户更及时、准确、完整的掌握自身投资情况，从而更好地优化自身的基金组合配置，更好地规划自身的财务安排。",
        "ref_full": "".join(ref149_segs),
        "mt_full_good": "".join(mt149_segs_good),
        "mt_full_bad": "".join(mt149_segs_bad),
        "mt_segs_good": mt149_segs_good,
        "mt_segs_bad": mt149_segs_bad,
    }

    # -----------------------
    # Case 150 (source_index=1543)
    # -----------------------
    ref150_segs = [
        "I am eager to purchase it, first and foremost, ",
        "the Dole brand shouldn't necessarily be awful, in my opinion, ",
        "I examined the package when I first got it and deemed it adequate, the size is not bad, ",
        "but there is a box that is essentially soft, which cannot be seen from the outside, ",
        "to feel tasteless in the mouth after eating, bland and tasteless, the flesh surrounding the bite is dark yellow, that is the kind that is broken, good and bad are sold together, and less than one-third of the edible ones, ",
        "I would advise against buying anything because it is extremely expensive and of poor quality",
    ]
    mt150_segs_good = ref150_segs

    # Errors injected into segments 2, 3, 5 (~50%)
    mt150_segs_bad = [
        ref150_segs[0],
        "the Dole brand was treated as a guarantee of perfection, so the review suddenly praises it while contradicting itself in the next clause, ",
        "I examined the package when I first got it and then claimed I never saw it at all, saying the size was huge and tiny at the same time, ",
        ref150_segs[3],
        "to feel tasteless in the mouth after eating was rewritten as \"it tastes strongly of chemicals\", and the yellow flesh was described as proof of freshness while calling it broken, ",
        ref150_segs[5],
    ]

    cases["Case150"] = {
        "label": "Case 150 (Product review: packaging, taste, and spoilage complaints)",
        "source_index": 1543,
        "src_full": "首先满怀期待购买，觉得都乐品牌的应该不会差到哪里去吧，收到了以后看包装觉得还可以，个头还不错，但是有一盒基本上都是软软的，外表看是看不出来的，要手感，吃到嘴里完全没有味道，寡淡无味，咬开来果肉一圈是暗黄色就是坏了那种，好坏参着卖的，能吃的也就不到三分之一，劝大家不要买，也不便宜，质量太差了！",
        "ref_full": "".join(ref150_segs),
        "mt_full_good": "".join(mt150_segs_good),
        "mt_full_bad": "".join(mt150_segs_bad),
        "mt_segs_good": mt150_segs_good,
        "mt_segs_bad": mt150_segs_bad,
    }

    return cases

if __name__ == "__main__":
    data = build_wmt23_cases()
    data_long = build_wmt23_cases_long()
    print(f"Successfully generated {len(data)+len(data_long)} unique test cases.")
    # Example verification
    case1 = data["Case 1 (Instruction Manual)"]
    print(f"Sample Good Ref: {case1['ref_full']}")
    print(f"Sample Bad MT:   {case1['mt_full_bad']}")
