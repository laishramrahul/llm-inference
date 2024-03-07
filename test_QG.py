import re
from transformers import TextStreamer
import torch
import random
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
peft_model_path="./peft-question-generation_ver2_result"
peft_model_dir = "peft-question-generation_ver2_result"
import time
custom_cache_directory = "./models"
import nltk 
nltk.download('punkt')
start_time = time.time()
# load base LLM model and tokenizer
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True, cache_dir=custom_cache_directory,
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir, cache_dir=custom_cache_directory)
def split_paragraph(paragraph, max_words):
    start_time = time.time()
    sentences = nltk.sent_tokenize(paragraph)
    result = []
    current_sentence = ""
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        
        if len(current_sentence.split()) + len(words) <= max_words:
            current_sentence += sentence + " "
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence + " "
    
    if current_sentence:
        result.append(current_sentence.strip())
    print("--- %s seconds ---" % (time.time() - start_time))
    return result

def generate_main(context):
  input=context       
  result = []
  prompt = f"""
  Generate Multiple choice question with one correct answer and three wrong options from the paragraph. Mark the correct answer.
  ### Input:
  {input}
  ### Question:
  """
  #print("the input\n",input)
  input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
  outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=1000,repetition_penalty=1.1 )
  output= tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
  output = "\n".join(line.lstrip() for line in output.splitlines())
  print("\nthe generated:\n",output)
  pattern = r'<\|QUESTION\|>(.*?)<\|/QUESTION\|>'
  match = re.search(pattern, output)
  if match:
     sentence = match.group(1)
     print("the sentence:",sentence)  
     pattern = r'<\|QUESTION\|>(.*?)<\|CORRECT\|>'
     match = re.search(pattern, output)
     question = match.group(1)
     #print("the question:",question)
     pattern = r'<\|CORRECT\|>(.*?)<\|/'
     match = re.search(pattern, output)
     correct_string = match.group(1)
     correct = correct_string
     #print("the correct answer:",correct_string)
     pattern = r'<\|INCORRECT1\|>(.*?)<\|/'
     match = re.search(pattern, output)
     wrong_option = match.group(1)
     option_ = [wrong_option]
     #print("the wrong option 1:",wrong_option)
     pattern = r'<\|INCORRECT2\|>(.*?)<\|/'
     match = re.search(pattern, output)
     wrong_option = match.group(1)
     option_.append(wrong_option)
     #print("the wrong option 2:",wrong_option)
     pattern = r'<\|INCORRECT3\|>(.*?)<\|/'
     match = re.search(pattern, output)
     wrong_option = match.group(1)
     option_.append(wrong_option)
     #print("the wrong option 3:",wrong_option)
     inter_dict = {}
     inter_dict["Question"] = question
     inter_dict["ChoicesFlag"] = True
     inter_dict["Ans"] = []  
     ans_dict = {}
     ans_dict["choices"] = correct
     ans_dict["ans_flag"] = True            
     inter_dict["Ans"].append(ans_dict)
     for element in option_:
         ans_dict = {}
         ans_dict["choices"] = element
         ans_dict["ans_flag"] = False
         inter_dict["Ans"].append(ans_dict)       
     random.shuffle(inter_dict["Ans"]) 
     #print([inter_dict])     
     return([inter_dict])
def generate(input):
    result=[]
    words = nltk.word_tokenize(input)
    if len(words)<400:
        input=' '.join(words)
        result = generate_main(input)
    else:
        
        max_words_per_sentence=400
        split=split_paragraph(input, max_words_per_sentence)
        #print(len(split))
        split=split[:5]
        for i, chunk in enumerate(split, start=1):
            #print("the chunks \n:",chunk)
            out = generate_main(chunk) 
            result.extend(out)
    return result
input = "The Himalayas consists of four parallel mountain ranges from south to north: the Sivalik Hills on the south; the Lower Himalayan Range; the Great Himalayas, which is the highest and central range; and the Tibetan Himalayas on the north.[16] The Karakoram are generally considered separate from the Himalayas.In the middle of the great curve of the Himalayan mountains lie the 8,000 m (26,000 ft) peaks of Dhaulagiri and Annapurna in Nepal, separated by the Kali Gandaki Gorge. The gorge splits the Himalayas into Western and Eastern sections, both ecologically and orographically – the pass at the head of the Kali Gandaki, the Kora La, is the lowest point on the ridgeline between Everest and K2 (the highest peak of the Karakoram range). To the east of Annapurna are the 8,000 m (5.0 mi) peaks of Manaslu and across the border in Tibet, Shishapangma. To the south of these lies Kathmandu, the capital of Nepal and the largest city in the Himalayas. East of the Kathmandu Valley lies the valley of the Bhote/Sun Kosi river which rises in Tibet and provides the main overland route between Nepal and China – the Araniko Highway/China National Highway 318. Further east is the Mahalangur Himal with four of the world's six highest mountains, including the highest: Cho Oyu, Everest, Lhotse, and Makalu. The Khumbu region, popular for trekking, is found here on the south-western approaches to Everest. The Arun river drains the northern slopes of these mountains, before turning south and flowing to the range to the east of Makalu.In the far east of Nepal, the Himalayas rise to the Kangchenjunga massif on the border with India, the third-highest mountain in the world, the most easterly 8,000 m (26,000 ft) summit and the highest point of India. The eastern side of Kangchenjunga is in the Indian state of Sikkim. Formerly an independent Kingdom, it lies on the main route from India to Lhasa, Tibet, which passes over the Nathu La pass into Tibet. East of Sikkim lies the ancient Buddhist Kingdom of Bhutan. The highest mountain in Bhutan is Gangkhar Puensum, which is also a strong candidate for the highest unclimbed mountain in the world. The Himalayas here are becoming increasingly rugged, with heavily forested steep valleys. The Himalayas continue, turning slightly northeast, through the Indian State of Arunachal Pradesh as well as Tibet, before reaching their easterly conclusion in the peak of Namche Barwa, situated in Tibet, inside the great bend of the Yarlang Tsangpo river. On the other side of the Tsangpo, to the east, are the Kangri Garpo mountains. The high mountains to the north of the Tsangpo, including Gyala Peri, however, are also sometimes included in the Himalayas.Going west from Dhaulagiri, Western Nepal is somewhat remote and lacks major high mountains, but is home to Rara Lake, the largest lake in Nepal. The Karnali River rises in Tibet but cuts through the centre of the region. Further west, the border with India follows the Sarda River and provides a trade route into China, where on the Tibetan plateau lies the high peak of Gurla Mandhata. Just across Lake Manasarovar from this lies the sacred Mount Kailash in the Kailash Ranges, which stands close to the source of the four main rivers of Himalayas and is revered in Hinduism, Jainism, Buddhism, Sufism and Bonpo. In Uttarakhand, the Himalayas are regionally divided into the Kumaon and Garhwal Himalayas with the high peaks of Nanda Devi and Kamet.[17] The state is also home to the important pilgrimage destinations of Chaar Dhaam, with Gangotri, the source of the holy river Ganges, Yamunotri, the source of the river Yamuna, and the temples at Badrinath and Kedarnath.The next Himalayan Indian state, Himachal Pradesh, is noted for its hill stations, particularly Shimla, the summer capital of the British Raj, and Dharamsala, the centre of the Tibetan community and government in exile in India. This area marks the start of the Punjab Himalaya and the Sutlej river, the most easterly of the five tributaries of the Indus, cuts through the range here. Further west, the Himalayas form much of the disputed Indian-administered union territory of Jammu and Kashmir where lie the mountainous Jammu region and the renowned Kashmir Valley with the town and lakes of Srinagar. The Himalayas form most of the south-west portion of the disputed Indian-administered union territory of Ladakh. The twin peaks of Nun Kun are the only mountains over 7,000 m (4.3 mi) in this part of the Himalayas. Finally, the Himalayas reach their western end in the dramatic 8000 m peak of Nanga Parbat, which rises over 8,000 m (26,000 ft) above the Indus valley and is the most westerly of the 8000 m summits. The western end terminates at a magnificent point near Nanga Parbat where the Himalayas intersect with the Karakoram and Hindu Kush ranges, in the disputed Pakistani-administered territory of Gilgit-Baltistan. Some portion of the Himalayas, such as the Kaghan Valley, Margalla Hills, and Galyat tract, extend into the Pakistani provinces of Khyber Pakhtunkhwa and Punjab. The Himalayan range is one of the youngest mountain ranges on the planet and consists mostly of uplifted sedimentary and metamorphic rock. According to the modern theory of plate tectonics, its formation is a result of a continental collision or orogeny along the convergent boundary (Main Himalayan Thrust) between the Indo-Australian Plate and the Eurasian Plate. The Arakan Yoma highlands in Myanmar and the Andaman and Nicobar Islands in the Bay of Bengal were also formed as a result of this collision.[19]During the Upper Cretaceous, about 70 million years ago, the north-moving Indo-Australian Plate (which has subsequently broken into the Indian Plate and the Australian Plate[20]) was moving at about 15 cm (5.9 in) per year. About 50 million years ago this fast-moving Indo-Australian Plate had completely closed the Tethys Ocean, the existence of which has been determined by sedimentary rocks settled on the ocean floor and the volcanoes that fringed its edges. Since both plates were composed of low density continental crust, they were thrust faulted and folded into mountain ranges rather than subducting into the mantle along an oceanic trench.[18] An often-cited fact used to illustrate this process is that the summit of Mount Everest is made of unmetamorphosed marine Ordovician limestone with fossil trilobites, crinoids, and ostracods from this ancient ocean.[21]Today, the Indian plate continues to be driven horizontally at the Tibetan Plateau, which forces the plateau to continue to move upwards.[22] The Indian plate is still moving at 67 mm (2.6 in) per year, and over the next 10 million years, it will travel about 1,500 km (930 mi) into Asia. About 20 mm per year of the India-Asia convergence is absorbed by thrusting along the Himalaya southern front. This leads to the Himalayas rising by about 5 mm per year, making them geologically active. The movement of the Indian plate into the Asian plate also makes this region seismically active, leading to earthquakes from time to time.[citation needed]During the last ice age, there was a connected ice stream of glaciers between Kangchenjunga in the east and Nanga Parbat in the west.[23][24] In the west, the glaciers joined with the ice stream network in the Karakoram, and in the north, they joined with the former Tibetan inland ice. To the south, outflow glaciers came to an end below an elevation of 1,000–2,000 m (3,300–6,600 ft).[23][25] While the current valley glaciers of the Himalaya reach at most 20 to 32 km (12 to 20 mi) in length, several of the main valley glaciers were 60 to 112 km (37 to 70 mi) long during the ice age.[23] The glacier snowline (the altitude where accumulation and ablation of a glacier are balanced) was about 1,400–1,660 m (4,590–5,450 ft) lower than it is today. Thus, the climate was at least 7.0 to 8.3 °C (12.6 to 14.9 °F) colder than it is today.[26]  The great ranges of central Asia, including the Himalayas, contain the third-largest deposit of ice and snow in the world, after Antarctica and the Arctic.[29] Some even refer to this region as the \"Third Pole.\"[30] The Himalayan range encompasses about 15,000 glaciers, which store about 12,000 km3 (2,900 cu mi), or 3600-4400 Gt (1012 kg)[30] of fresh water.[31] Its glaciers include the Gangotri and Yamunotri (Uttarakhand) and Khumbu glaciers (Mount Everest region), Langtang glacier (Langtang region), and Zemu (Sikkim).Owing to the mountains' latitude near the Tropic of Cancer, the permanent snow line is among the highest in the world, at typically around 5,500 m (18,000 ft).[32] In contrast, equatorial mountains in New Guinea, the Rwenzoris, and Colombia have a snow line some 900 m (2,950 ft) lower.[33] The higher regions of the Himalayas are snowbound throughout the year, in spite of their proximity to the tropics, and they form the sources of several large perennial rivers. In recent years, scientists have monitored a notable increase in the rate of glacier retreat across the region as a result of climate change.[34][35] For example, glacial lakes have been forming rapidly on the surface of debris-covered glaciers in the Bhutan Himalaya during the last few decades. Studies have measured an approximately 13% overall decrease in glacial coverage in the Himalayas over the last 40–50 years.[30] Local conditions play a large role in glacial retreat, however, and glacial loss can vary locally from a few m/yr to 61 m/yr.[30] A marked acceleration in glacial mass loss has also been observed since 1975, from about 5-13 Gt/yr to 16-24 Gt/yr.[30] Although the effect of this will not be known for many years, it potentially could mean disaster for the hundreds of millions of people who rely on the glaciers to feed the rivers during the dry seasons.[30][36][37][38] The global climate change will affect the water resources and livelihoods of the Greater Himalayan region.[citation needed]The Himalayan region is dotted with hundreds of lakes.[39] Pangong Tso, which is spread across the border between India and China, at the far western end of Tibet, is among the largest with a surface area of 700 km2 (270 sq mi).South of the main range, the lakes are smaller. Tilicho Lake in Nepal, in the Annapurna massif, is one of the highest lakes in the world. Other lakes include Rara Lake in western Nepal, She-Phoksundo Lake in the Shey Phoksundo National Park of Nepal, Gurudongmar Lake, in North Sikkim, Gokyo Lakes in Solukhumbu district of Nepal, and Lake Tsongmo, near the Indo-China border in Sikkim.[39]Some of the lakes present the danger of a glacial lake outburst flood. The Tsho Rolpa glacier lake in the Rowaling Valley, in the Dolakha District of Nepal, is rated as the most dangerous. The lake, which is located at an altitude of 4,580 m (15,030 ft), has grown considerably over the last 50 years due to glacial melting.[40][41] The mountain lakes are known to geographers as tarns if they are caused by glacial activity. Tarns are found mostly in the upper reaches of the Himalaya, above 5,500 m (18,000 ft).[42]Temperate Himalayan wetlands provide important habitat and layover sites for migratory birds. Many mid and low altitude lakes remain poorly studied in terms of their hydrology and biodiversity, like Khecheopalri in the Sikkim Eastern Himalayas.[43] The physical factors determining the climate in any location in the Himalayas include latitude, altitude, and the relative motion of the Southwest monsoon.[44] From north to south, the mountains cover more than eight degrees of latitude, spanning temperate to subtropical zones.[44] The colder air of Central Asia is prevented from blowing down into South Asia by the physical configuration of the Himalayas.[44] This causes the tropical zone to extend farther north in South Asia than anywhere else in the world.[44] The evidence is unmistakable in the Brahmaputra valley as the warm air from the Bay of Bengal bottlenecks and rushes up past Namcha Barwa, the eastern anchor of the Himalayas, and into southeastern Tibet.[44] Temperatures in the Himalayas cool by 2.0 degrees C (3.6 degrees F) for every 300 metres (980 ft) increase of altitude.[44]Gandaki River in NepalAs the physical features of mountains are irregular, with broken jagged contours, there can be wide variations in temperature over short distances.[45] Temperature at a location on a mountain depends on the season of the year, the bearing of the sun with respect to the face on which the location lies, and the mass of the mountain, i.e. the amount of matter in the mountain.[45] As the temperature is directly proportional to received radiation from the sun, the faces that receive more direct sunlight also have a greater heat buildup.[45] In narrow valleys—lying between steep mountain faces—there can be dramatically different weather along their two margins.[45] The side to the north with a mountain above facing south can have an extra month of the growing season.[45] The mass of the mountain also influences the temperature, as it acts as a heat island, in which more heat is absorbed and retained than the surroundings, and therefore influences the heat budget or the amount of heat needed to raise the temperature from the winter minimum to the summer maximum.[45] The immense scale of the Himalayas means that many summits can create their own weather, the temperature fluctuating from one summit to another, from one face to another, and all may be quite different from the weather in nearby plateaus or valleys.[45]A critical influence on the Himalayan climate is the Southwest Monsoon. This is not so much the rain of the summer months as the wind that carries the rain.[45] Different rates of heating and cooling between the Central Asian continent and the Indian Ocean create large differences in the atmospheric pressure prevailing above each.[45] In the winter, a high-pressure system forms and remains suspended above Central Asia, forcing air to flow in the southerly direction over the Himalayas.[45] But in Central Asia, as there is no substantial source for water to be diffused as vapour, the winter winds blowing across South Asia are dry.[45] In the summer months, the Central Asian plateau heats up more than the ocean waters to its south. As a result, the air above it rises higher and higher, creating a thermal low.[45] Off-shore high-pressure systems in the Indian Ocean push the moist summer air inland toward the low-pressure system. When the moist air meets mountains, it rises and upon subsequent cooling, its moisture condenses and is released as rain, typically heavy rain.[45] The wet summer monsoon winds cause precipitation in India and all along the layered southern slopes of the Himalayas. This forced lifting of air is called the orographic effect.[45]The vast size, huge altitude range, and complex topography of the Himalayas mean they experience a wide range of climates, from humid subtropical in the foothills, to cold and dry desert conditions on the Tibetan side of the range. For much of the Himalayas—in the areas to the south of the high mountains, the monsoon is the most characteristic feature of the climate and causes most of the precipitation, while the western disturbance brings winter precipitation, especially in the west. Heavy rain arrives on the southwest monsoon in June and persists until September. The monsoon can seriously impact transport and cause major landslides. It restricts tourism – the trekking and mountaineering season is limited to either before the monsoon in April/May or after the monsoon in October/November (autumn). In Nepal and Sikkim, there are often considered to be five seasons: summer, monsoon, autumn, (or post-monsoon), winter, and spring.[citation needed]Using the Köppen climate classification, the lower elevations of the Himalayas, reaching in mid-elevations in central Nepal (including the Kathmandu valley), are classified as Cwa, Humid subtropical climate with dry winters. Higher up, most of the Himalayas have a subtropical highland climate (Cwb).[citation needed]The intensity of the southwest monsoon diminishes as it moves westward along the range, with as much as 2,030 mm (80 in) of rainfall in the monsoon season in Darjeeling in the east, compared to only 975 mm (38.4 in) during the same period in Shimla in the west.[46][47]The northern side of the Himalayas, also known as the Tibetan Himalaya, is dry, cold, and generally windswept, particularly in the west where it has a cold desert climate. The vegetation is sparse and stunted and the winters are severely cold. Most of the precipitation in the region is in the form of snow during the late winter and spring months.The cold desert region of Upper Mustang; the region lies to the north of the Annapurna massif (visible in the background)A village in the Pokhara Valley during the monsoon season; the valley lies to the south of the Annapurna massifLocal impacts on climate are significant throughout the Himalayas. Temperatures fall by 0.2 to 1.2 °C for every 100 m (330 ft) rise in altitude.[48] This gives rise to a variety of climates, from a nearly tropical climate in the foothills, to tundra and permanent snow and ice at higher elevations. Local climate is also affected by the topography: The leeward side of the mountains receive less rain while the well-exposed slopes get heavy rainfall and the rain shadow of large mountains can be significant, for example, leading to near desert conditions in the Upper Mustang, which is sheltered from the monsoon rains by the Annapurna and Dhaulagiri massifs and has annual precipitation of around 300 mm (12 in), while Pokhara on the southern side of the massifs has substantial rainfall (3,900 mm or 150 in a year). Thus, although annual precipitation is generally higher in the east than in the west, local variations are often more important.[citation needed]The Himalayas have a profound effect on the climate of the Indian subcontinent and the Tibetan Plateau. They prevent frigid, dry winds from blowing south into the subcontinent, which keeps South Asia much warmer than corresponding temperate regions in the other continents. It also forms a barrier for the monsoon winds, keeping them from traveling northwards, and causing heavy rainfall in the Terai region. The Himalayas are also believed to play an important part in the formation of Central Asian deserts, such as the Taklamakan and Gobi.[49]"
#input="The British brought cricket to India in the early 1700s, with the first cricket match played in 1721.[15] It was played and adopted by Kolis of Gujarat because they were sea pirates and outlaws who always loot the British ships so East India Company tried to manage the Kolis in cricket and been successful.[16][17][18] In 1848, the Parsi community in Mumbai formed the Oriental Cricket Club, the first cricket club to be established by Indians. After slow beginnings, the Europeans eventually invited the Parsis to play a match in 1877.[19] By 1912, the Parsis, Hindus, Sikhs and Muslims of Bombay played a quadrangular tournament with the Europeans every year.[19] In the early 1900s, some Indians went on to play for the England cricket team. Some of these, such as Ranjitsinhji and Duleepsinhji were greatly appreciated by the British and their names went on to be used for the Ranji Trophy and Duleep Trophy – two major first-class tournaments in India. In 1911, an Indian men's cricket team, captained by Bhupinder Singh of Patiala, went on their first official tour of the British Isles, but only played English county teams and not the England cricket team.[20][21]"
words = nltk.word_tokenize(input)
if len(words)<400:
  input=' '.join(words)
  generate(input)
else:
  result=[]
  max_words_per_sentence=400
  split=split_paragraph(input, max_words_per_sentence)
  #print(len(split))
  split=split[:5]
  for i, chunk in enumerate(split, start=1):
    #print("the chunks \n:",chunk)
    generate(chunk)
#print("--- %s seconds ---" % (time.time() - start_time))