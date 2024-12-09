import os
import re
import sys
import json
import random
import logging
import argparse

import openai
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(funcName)s() - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def read_txt(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        text = f.read()

    if write_log:
        characters = len(text)
        logger.info(f"Read {characters:,} characters")
    return text


def write_txt(file, text, write_log=False):
    if write_log:
        characters = len(text)
        logger.info(f"Writing {characters:,} characters to {file}")

    with open(file, "w", encoding="utf8") as f:
        f.write(text)

    if write_log:
        logger.info(f"Written")
    return


def read_json(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


def write_json(file, data, indent=None, write_log=False):
    if write_log:
        objects = len(data)
        logger.info(f"Writing {objects:,} objects to {file}")

    with open(file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    if write_log:
        logger.info(f"Written")
    return


class Config:
    def __init__(self, config_file):
        data = read_json(config_file)

        self.model = data["model"]
        self.static_dir = os.path.join(*data["static_dir"])
        self.state_dir = os.path.join(*data["state_dir"])

        os.makedirs(self.state_dir, exist_ok=True)
        return


class GPT:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.model_candidate_tokens = {
            "gpt-3.5-turbo": {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
            },
            "gpt-4": {
                "gpt-4": 8192,
                "gpt-4-32k": 32768,
            }
        }
        return

    def get_specific_tokens_model(self, text_in, out_tokens):
        in_token_list = self.tokenizer.encode(text_in)
        in_tokens = len(in_token_list)
        tokens = in_tokens + out_tokens

        for candidate, max_tokens in self.model_candidate_tokens.get(self.model, {}).items():
            if max_tokens >= tokens:
                break
        else:
            candidate = ""

        return in_tokens, candidate

    def run_gpt(self, text_in, out_tokens):
        in_tokens, specific_tokens_model = self.get_specific_tokens_model(text_in, out_tokens)
        if not specific_tokens_model:
            return ""

        # logger.info(text_in)
        logger.info("I'm alive! Please wait awhile >< ...")

        completion = openai.ChatCompletion.create(
            model=specific_tokens_model,
            n=1,
            messages=[
                {"role": "user", "content": text_in},
            ]
        )

        text_out = completion.choices[0].message.content
        return text_out


class State:
    def __init__(self, save_file=""):
        self.save_file = save_file
        self.log = ""
        self.level = 1
        self.exp = 0
        self.unlocked_opponents = ["Pidgey"]
        self.unlocked_dex = []
        self.pikachu = {
            "name": "皮卡丘",
            "current_hp": 101,
            "max_hp": 101,
            "attack": 20,
            "friendliness": 0,
            "moves": [
                {"name": "電擊", "type": "電", "damage": 10},
            ],
        }
        self.dex = {}
        self.ended = False

    def save(self):
        data = {
            "log": self.log,
            "level": self.level,
            "exp": self.exp,
            "unlocked_opponents": self.unlocked_opponents,
            "unlocked_dex": self.unlocked_dex,
            "pikachu": self.pikachu,
            "dex": self.dex,
            "ended": self.ended,
        }
        if self.save_file:
            with open(self.save_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    def load(self):
        if self.save_file and os.path.exists(self.save_file):
            with open(self.save_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.log = data.get("log", "")
                self.level = data.get("level", 1)
                self.exp = data.get("exp", 0)
                self.unlocked_opponents = data.get("unlocked_opponents", ["Pidgey"])
                self.unlocked_dex = data.get("unlocked_dex", [])
                self.pikachu = data.get("pikachu", {
                    "name": "皮卡丘",
                    "current_hp": 101,
                    "max_hp": 101,
                    "attack": 20,
                    "friendliness": 0,
                    "moves": [
                        {"name": "電擊", "type": "電", "damage": 10},
                    ],
                })
                self.dex = data.get("dex", {})
                self.ended = data.get("ended", False)

class Game:
    def __init__(self, config):
        self.static_dir = config.static_dir
        self.state_dir = config.state_dir
        self.summary_file = ""
        self.gpt = GPT(config.model)
        self.gpt4 = GPT("gpt-4")

        self.user_prompt_to_text = {}
        self.max_saves = 4
        self.state = State()
        self.all_pokemon_list = [
            "Pidgey", "Squirtle", "Charizard", "Dragonite", "Shaymin", "Arceus",
        ]
        self.action_to_text = {
            "home": "回家(退出遊戲)",
            "pokedex": "觀看寶可夢圖鑑",
            "battle": "尋找對戰對手",
            "play_with_pikachu": "與皮卡丘玩耍",
            "heal_pikachu": "治療皮卡丘",
        }
        self.pokemon_data = {}
        pokemon_dir = os.path.join(self.static_dir, "pokemon")
        for filename in os.listdir(pokemon_dir):
            pokemon_name = filename[:-5]  # Remove '.json'
            pokemon_file = os.path.join(pokemon_dir, filename)
            self.pokemon_data[pokemon_name] = read_json(pokemon_file)

        self.type_effectiveness = read_json(os.path.join(self.static_dir, "type_effectiveness_config.json"))
        self.max_level = 100

        # Load prompt text
        user_prompt_dir = os.path.join(self.static_dir, "user_prompt")
        for filename in os.listdir(user_prompt_dir):
            user_prompt = filename[:-4]
            user_prompt_file = os.path.join(user_prompt_dir, filename)
            self.user_prompt_to_text[user_prompt] = read_txt(user_prompt_file)

    def run_start(self):
        user_prompt = self.user_prompt_to_text["start"]
        while True:
            text_in = input(user_prompt)
            if text_in == "1":
                start_type = "new"
                break
            elif text_in == "2":
                start_type = "load"
                break

        save_list_text = "\n存檔列表：\n"
        saveid_to_exist = {}
        for i in range(self.max_saves):
            save_id = str(i + 1)
            save_file = os.path.join(self.state_dir, f"save_{save_id}.json")
            if os.path.exists(save_file):
                saveid_to_exist[save_id] = True
                # 讀取存檔基本資訊
                with open(save_file, "r", encoding="utf-8") as f:
                    save_data = json.load(f)
                    level = save_data.get("level", 1)
                    exp = save_data.get("exp", 0)
                    exp_to_next = 10 + (level - 1) * 5
                    friendliness = save_data.get("pikachu", {}).get("friendliness", 0)
                    ended = save_data.get("ended", False)

                finished_text = " finished!!!" if ended else ""
                save_list_text += (
                    f"({save_id}) 舊有存檔：★ 當前等級：{level}, 經驗值：{exp}/{exp_to_next}, "
                    f"好感度：{friendliness} ★ {finished_text}\n"
                )
            else:
                saveid_to_exist[save_id] = False
                save_list_text += f"({save_id}) 空白存檔\n"

        user_prompt = f"{save_list_text}\n使用存檔欄位： "
        while True:
            text_in = input(user_prompt)
            if start_type == "new" and text_in in saveid_to_exist:
                use_save_id = text_in
                break
            elif start_type == "load" and saveid_to_exist.get(text_in, False):
                use_save_id = text_in
                # 檢查是否為已完成的存檔
                save_file = os.path.join(self.state_dir, f"save_{use_save_id}.json")
                with open(save_file, "r", encoding="utf-8") as f:
                    save_data = json.load(f)
                    if save_data.get("ended", False):
                        finished_file = os.path.join(self.static_dir, "user_prompt", "finished.txt")
                        if os.path.exists(finished_file):
                            finished_text = read_txt(finished_file)
                            print("\n這個存檔已完成：")
                            print(finished_text)
                        else:
                            print("\n這個存檔已完成，但未找到完成訊息檔案！")
                        print("\n請選擇其他存檔或重新開始。")
                        continue
                break

        use_save_file = os.path.join(self.state_dir, f"save_{use_save_id}.json")
        self.state = State(use_save_file)

        if start_type == "new":
            user_prompt = self.user_prompt_to_text["opening"]
            input(user_prompt + "\n與皮卡丘的旅程開始了(按換行繼續)... ")
            self.state.log += user_prompt
            self.state.level = 1
            self.state.exp = 0
            self.state.unlocked_opponents = ["Pidgey"]
            self.state.unlocked_dex = []
            self.state.pikachu = {
                "name": "皮卡丘",
                "current_hp": 101,
                "max_hp": 101,
                "attack": 20,
                "friendliness": 0,
                "moves": [
                    {"name": "電擊", "type": "電", "damage": 10},
                ],
            }
            self.state.save()
        else:
            self.state.load()

        self.run_loop()

    def get_action(self):
        print(f"\n★ 當前等級：{self.state.level}, 經驗值：{self.state.exp}/{self.exp_to_next_level()}, 好感度：{self.state.pikachu['friendliness']} ★")
        action_list = ["home", "pokedex", "battle", "play_with_pikachu"]

        if self.state.pikachu["current_hp"] < self.state.pikachu["max_hp"]:
            action_list.append("heal_pikachu")

        action_list_text = "\n行動列表：\n"
        actionid_to_action = {}
        for i, action in enumerate(action_list):
            action_id = str(i)
            actionid_to_action[action_id] = action
            action_text = self.action_to_text[action]
            action_list_text += f"({action_id}) {action_text}\n"

        user_prompt = f"{action_list_text}\n你的下一步： "
        while True:
            text_in = input(user_prompt)
            if text_in in actionid_to_action:
                return actionid_to_action[text_in]

    def is_game_finished(self):
        """
        Check if the game-ending conditions are met.
        The game ends when Pikachu reaches level 100 and has at least 50 friendliness.
        """
        return self.state.level >= 100 and self.state.pikachu["friendliness"] >= 50

    def run_loop(self):
        while True:
            if self.is_game_finished():
                print("\n恭喜！你和皮卡丘完成了旅程，達成了遊戲目標！")
                ending_file = os.path.join(self.static_dir, "user_prompt", "ending.txt")
                ending_text = read_txt(ending_file)
                print(ending_text)
                self.state.ended = True
                self.state.save()
                sys.exit()

            action = self.get_action()

            if action == "home":
                print("正在存檔，請稍候...")
                self.state.save()
                print("遊戲已成功存檔！")
                print("皮卡丘很開心，期待下次一起冒險！")
                sys.exit()

            elif action == "pokedex":
                self.do_pokedex()

            elif action == "battle":
                self.do_battle()

            elif action == "play_with_pikachu":
                self.do_play_with_pikachu()

            elif action == "heal_pikachu":
                self.do_heal_pikachu()

    def do_pokedex(self):
        a = len(self.state.unlocked_dex)
        print(f"\n=========================================================================")
        if a == 0:
            print(f"\n目前還沒有遇到過任何寶可夢！")
        else:
            print("\n寶可夢圖鑑：")
            for name in self.state.unlocked_dex:
                if name in self.pokemon_data:
                    self.display(name)
                if a >= 2:
                    print(f"\n=========================================================================")
                    a -= 1
        print(f"\n=========================================================================")
        input("\n(按換行繼續)...")

    def display(self, name):
        data = self.pokemon_data[name]
        print(f"\n{data['name']}:")
        print(f"屬性: {data['type']}")
        print(f"介紹: {data['description']}")
        print("招式:")
        for move in data["moves"]:
            print(f"  - 名稱: {move['name']}, 屬性: {move['type']}, 傷害: {move['damage']}")
        print(f"血量: {data['hp']}")

    def do_battle(self):
        print("\n你開始尋找對戰對手！")
        opponent_name = random.choice(self.state.unlocked_opponents)
        opponent = self.pokemon_data[opponent_name].copy()
        opponent["current_hp"] = opponent["hp"]

        print(f"你遇到了對手的寶可夢：{opponent['name']}！")
        print(f"皮卡丘 血量：{self.state.pikachu['current_hp']}/{self.state.pikachu['max_hp']}")

        while self.state.pikachu["current_hp"] > 0 and opponent["current_hp"] > 0:
            # 玩家回合
            print("\n你的回合！")
            print("皮卡丘可以使用的招式：")
            for i, move in enumerate(self.state.pikachu["moves"], start=1):
                print(f"({i}) {move['name']} (屬性: {move['type']}, 傷害: {move['damage']})")

            # 讓玩家選擇招式
            while True:
                try:
                    choice = int(input("選擇一個招式（輸入數字）："))
                    if 1 <= choice <= len(self.state.pikachu["moves"]):
                        selected_move = self.state.pikachu["moves"][choice - 1]
                        break
                    else:
                        print("無效的選擇，請重新輸入。")
                except ValueError:
                    print("請輸入有效的數字。")

            base_damage = selected_move["damage"]
            damage = self.calculate_damage(selected_move["type"], opponent["type"], base_damage)
            opponent["current_hp"] -= damage
            print(f"\n皮卡丘使用了 {selected_move['name']}！對 {opponent['name']} 造成了 {damage} 點傷害。")
            print(f"{opponent['name']} 剩餘血量：{max(0, opponent['current_hp'])}")

            if opponent["current_hp"] <= 0:
                print(f"\n你擊敗了 {opponent['name']}！")
                self.gain_exp(opponent["exp"])
                break

            # 對手回合
            move = random.choice(opponent["moves"])
            base_damage = move["damage"]
            damage = self.calculate_damage(move["type"], "電", base_damage)
            self.state.pikachu["current_hp"] -= damage
            print(f"\n對手的回合！")
            print(f"{opponent['name']} 使用了技能：{move['name']}！（屬性：{move['type']}，威力：{move['damage']}）")
            print(f"對皮卡丘造成了 {damage} 點傷害。")
            print(f"皮卡丘剩餘血量：{max(0, self.state.pikachu['current_hp'])}")

            if self.state.pikachu["current_hp"] <= 0:
                print("\n你輸了對戰，下次再努力吧！")
                self.gain_exp(opponent["exp"] // 10)
                break

        if opponent_name not in self.state.unlocked_dex:
            self.state.unlocked_dex.append(opponent_name)

        input("\n(按換行繼續)...")


    def calculate_damage(self, attacker_type, defender_type, base_damage):
        effectiveness = self.type_effectiveness.get(attacker_type, {}).get(defender_type, 1)
        return int(base_damage * effectiveness)

    def gain_exp(self, exp):
        """Increase experience and handle leveling up."""
        self.state.exp += exp
        print(f"你獲得了 {exp} 點經驗值！")

        while self.state.exp >= self.exp_to_next_level():
            self.state.exp -= self.exp_to_next_level()
            self.level_up()

    def exp_to_next_level(self):
        """Calculate experience required to level up."""
        return 10 + (self.state.level - 1) * 5

    def level_up(self):
        """Level up Pikachu and unlock new moves."""
        self.state.level += 1
        self.state.pikachu["max_hp"] += 1
        self.state.pikachu["current_hp"] = self.state.pikachu["max_hp"]
        print(f"皮卡丘升級了！目前等級：{self.state.level}，最大血量：{self.state.pikachu['max_hp']}。")

        # Unlock new moves at specific levels
        new_moves = {
            20: {"name": "電光一閃", "type": "一般", "damage": 30},
            50: {"name": "十萬伏特", "type": "電", "damage": 50},
            70: {"name": "鐵尾", "type": "鋼", "damage": 70},
        }
        if self.state.level in new_moves:
            move = new_moves[self.state.level]
            self.state.pikachu["moves"].append(move)
            print(f"皮卡丘解鎖了新招式：{move}！")

        new_opo = {
            10: "Squirtle",
            30: "Charizard",
            40: "Dragonite",
            60: "Shaymin",
            80: "Arceus",
        }
        if self.state.level in new_opo:
            self.state.unlocked_opponents.append(new_opo[self.state.level])
            print(f"\n可能會遇到更厲害的寶可夢...")

    def do_play_with_pikachu(self):
        print("\n你決定和皮卡丘互動。")
        interaction = input("你想要做什麼？(例如：餵食、撫摸、玩耍)：")
        if len(interaction.strip()) > 0:
            response = self.gpt.run_gpt(f"我和皮卡丘進行了以下互動：{interaction}，皮卡丘是什麼反應？又做了什麼呢？", 50)
            print(f"{response}")
            self.state.pikachu["friendliness"] += 10
            print(f"你與皮卡丘的好感度增加了！目前好感度為：{self.state.pikachu['friendliness']}。")
        else:
            print("你什麼都沒做，皮卡丘似乎有點失望。")
        input("\n(按換行繼續)...")

    def do_heal_pikachu(self):
        print(f"\n皮卡丘當前血量：{self.state.pikachu['current_hp']}/{self.state.pikachu['max_hp']}")
        if self.state.pikachu["current_hp"] < self.state.pikachu["max_hp"]:
            restored_hp = self.state.pikachu["max_hp"] - self.state.pikachu["current_hp"]
            self.state.pikachu["current_hp"] = self.state.pikachu["max_hp"]
            print(f"\n皮卡丘受傷了，你幫它治療。恢復了 {restored_hp} 點血量！")
            self.state.pikachu["friendliness"] += 1
            print(f"你與皮卡丘的好感度增加了！目前好感度為：{self.state.pikachu['friendliness']}。")
        else:
            print("\n皮卡丘現在是滿血狀態，不需要治療！")
        input("\n(按換行繼續)...")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="pikachu_game_config.json")
    arg = parser.parse_args()

    openai.api_key = input("OpenAI API Key: ")

    config = Config(arg.config_file)
    game = Game(config)
    game.run_start()
    return


if __name__ == "__main__":
    main()
    sys.exit()
