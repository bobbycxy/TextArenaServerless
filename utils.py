import trueskill
from typing import Dict, List
from supabase import Client
import datetime
import json
import logging

def update_trueskill_scores(environment_id: int, rewards: Dict[int, float], player_dict: Dict[int, Dict], supabase: Client) -> Dict:
    """
    Calculate and update TrueSkill ratings for players based on game results.
    """
    result = {}
    player_ratings = {}
    human_tokens = set()
    
    for player_id in player_dict:
        token = player_dict[player_id]["token"]
        human_check = supabase.table("humans").select("id").eq("cookie_id", token).execute()
        is_human = len(human_check.data) > 0
        
        if is_human:
            human_tokens.add(token)
            model_id = 0
        else:
            model_response = supabase.table("models").select("id").eq("model_token", token).execute()
            if not model_response.data:
                raise ValueError(f"No model found for token: {token}")
            model_id = model_response.data[0]["id"]
        
        ts_response = supabase.table("trueskill").select("trueskill", "sd", "updated_at").eq("model_id", model_id).eq("environment_id", environment_id).order("updated_at", desc=True).execute()
        
        # Enhanced logging for trueskill data
        logging.info(f"TrueSkill response for player_id={player_id}, model_id={model_id}, environment_id={environment_id}:")
        logging.info(f"Full ts_response data: {json.dumps(ts_response.data, indent=2) if ts_response.data else 'No data'}")
        
        if ts_response.data:
            logging.info(f"Retrieved trueskill value: {ts_response.data[0]['trueskill']}, SD: {ts_response.data[0]['sd']}, Updated at: {ts_response.data[0]['updated_at']}")
            if len(ts_response.data) > 1:
                logging.info(f"Multiple records found - using most recent. Total records: {len(ts_response.data)}")
                for i, record in enumerate(ts_response.data):
                    logging.info(f"  Record {i+1}: Trueskill={record['trueskill']}, SD={record['sd']}, Updated at={record['updated_at']}")
        else:
            logging.info("No existing trueskill record found, will use default values (25.0, 8.333)")
        
        current_ts = ts_response.data[0]["trueskill"] if ts_response.data else 25.0
        current_sd = ts_response.data[0]["sd"] if ts_response.data else 8.333
        
        player_ratings[player_id] = {
            "token": token,
            "model_id": model_id,
            "rating": trueskill.Rating(mu=current_ts, sigma=current_sd)
        }
        result[token] = {
            "model_id": model_id,
            "old_trueskill": current_ts,
            "old_sd": current_sd
        }
    
    env_response = supabase.table("environments").select("game_type").eq("id", environment_id).execute()
    if not env_response.data:
        raise ValueError(f"No environment found with ID: {environment_id}")
    game_type = env_response.data[0]["game_type"]
    
    new_ratings = {}
    if game_type == "two_player":
        p1_id, p2_id = list(player_ratings.keys())
        p1_rating = player_ratings[p1_id]["rating"]
        p2_rating = player_ratings[p2_id]["rating"]
        if rewards[p1_id] > rewards[p2_id]:
            new_p1_rating, new_p2_rating = trueskill.rate_1vs1(p1_rating, p2_rating)
        elif rewards[p1_id] < rewards[p2_id]:
            new_p1_rating, new_p2_rating = trueskill.rate_1vs1(p2_rating, p1_rating)
        else:
            new_p1_rating, new_p2_rating = trueskill.rate_1vs1(p1_rating, p2_rating, drawn=True)
        new_ratings[p1_id] = new_p1_rating
        new_ratings[p2_id] = new_p2_rating
    elif game_type == "multi_player":
        ranked_players = sorted(player_ratings.keys(), key=lambda x: rewards[x], reverse=True)
        rating_groups = [[player_ratings[pid]["rating"]] for pid in ranked_players]
        new_rating_groups = trueskill.rate(rating_groups)
        for i, pid in enumerate(ranked_players):
            new_ratings[pid] = new_rating_groups[i][0]
    else:
        raise ValueError(f"Unsupported game type: {game_type}")
    
    # Log all trueskill changes
    logging.info("TrueSkill updates summary:")
    
    for player_id, new_rating in new_ratings.items():
        token = player_ratings[player_id]["token"]
        model_id = player_ratings[player_id]["model_id"]
        old_ts = result[token]["old_trueskill"]
        old_sd = result[token]["old_sd"]
        
        # Calculate the changes
        ts_change = new_rating.mu - old_ts
        sd_change = new_rating.sigma - old_sd
        
        result[token].update({
            "new_trueskill": new_rating.mu,
            "new_sd": new_rating.sigma,
            "trueskill_change": ts_change,
            "sd_change": sd_change
        })
        
        # Log before updating Supabase with clear change indicators
        logging.info(f"Player {player_id} (token={token}, model_id={model_id}):")
        logging.info(f"  - Trueskill: {old_ts:.2f} → {new_rating.mu:.2f} (Δ: {ts_change:+.2f})")
        logging.info(f"  - SD: {old_sd:.2f} → {new_rating.sigma:.2f} (Δ: {sd_change:+.2f})")
        
        # Perform the update
        upsert_response = supabase.table("trueskill").upsert({
            "model_id": model_id,
            "environment_id": environment_id,
            "trueskill": new_rating.mu,
            "sd": new_rating.sigma
        }).execute()
        
        # Log the response from the upsert operation
        logging.info(f"Supabase upsert response for player {player_id}: {upsert_response.data if hasattr(upsert_response, 'data') else 'No data'}")
    
    return result

def determine_outcomes(rewards: Dict[int, float]) -> Dict[int, str]:
    """
    Determines the outcome (win/loss/draw) for each player based on their rewards.
    """
    max_reward = max(rewards.values())
    num_max = sum(1 for r in rewards.values() if r == max_reward)
    outcomes = {}
    for pid, r in rewards.items():
        if r == max_reward:
            outcomes[pid] = "draw" if num_max > 1 else "win"
        else:
            outcomes[pid] = "loss"
    return outcomes

async def update_game_state(game_obj, rewards: Dict[int, float], reason: str, supabase: Client):
    """
    Updates Supabase tables (games, player_games, moves) and sends closing messages to players.
    Handles disconnections for non-human players by counting them as a loss.
    """
    logging.info("=== GAME STATE UPDATE ===")
    logging.info(f"Rewards: {rewards}")
    logging.info(f"Reason: {reason}")
    for pid in game_obj.player_dict:
        if not game_obj.connected[pid]:
            token = game_obj.player_dict[pid]["token"]
            human_check = supabase.table("humans").select("id").eq("cookie_id", token).execute()
            is_human = len(human_check.data) > 0
            if not is_human:
                if len(game_obj.player_dict) == 2:
                    other_pid = [p for p in game_obj.player_dict if p != pid][0]
                    rewards[pid] = -1
                    rewards[other_pid] = 1
                else:
                    rewards[pid] = -float('inf')

    outcomes = determine_outcomes(rewards)
    game_data = {"environment_id": game_obj.environment_id, "status": "finished", "reason": reason}
    game_response = supabase.table("games").insert(game_data).execute()
    game_id = game_response.data[0]["id"]

    logging.info(f"Game finished (id={game_id}). Calculating trueskill updates for all players.")
    
    trueskill_update_dict = update_trueskill_scores(
        environment_id=game_obj.environment_id,
        rewards=rewards,
        player_dict=game_obj.player_dict,
        supabase=supabase
    )

    logging.info(f"Trueskill updates completed. Recording player_games and moves to database.")
    
    player_game_ids = {}
    for pid in game_obj.player_dict:
        token = game_obj.player_dict[pid]["token"]
        model_id = trueskill_update_dict[token]["model_id"]
        human_id = token if model_id == 0 else None
        reward = rewards[pid]
        outcome = outcomes[pid]
        trueskill_change = trueskill_update_dict[token]["new_trueskill"] - trueskill_update_dict[token]["old_trueskill"]
        sd_change = trueskill_update_dict[token]["new_sd"] - trueskill_update_dict[token]["old_sd"]
        player_game_data = {
            "game_id": game_id,
            "model_id": model_id,
            "human_id": human_id,
            "player_id": pid,
            "reward": reward,
            "outcome": outcome,
            "trueskill_change": trueskill_change,
            "sd_change": sd_change,
            "env_id": game_obj.environment_id
        }
        player_game_response = supabase.table("player_games").insert(player_game_data).execute()
        player_game_ids[pid] = player_game_response.data[0]["id"]

    for pid in game_obj.player_dict:
        token = game_obj.player_dict[pid]["token"]
        player_game_id = player_game_ids[pid]
        for move in game_obj.moves[token]:
            move_data = {
                "game_id": game_id,
                "player_game_id": player_game_id,
                "observation": move["observation"],
                "timestamp_observation": datetime.datetime.fromtimestamp(move["timestamp_observation"]).isoformat(),
                "action": move["action"],
                "timestamp_action": datetime.datetime.fromtimestamp(move["timestamp_action"]).isoformat() if move["timestamp_action"] else None
            }
            supabase.table("moves").insert(move_data).execute()

    # Send game over messages and track their delivery
    message_send_exceptions = []
    
    for pid in game_obj.player_dict:
        websocket = game_obj.player_dict[pid]["websocket"]
        if websocket:
            try:
                token = game_obj.player_dict[pid]["token"]
                opponents = [{"player_id": other_pid, "outcome": outcomes[other_pid]} for other_pid in game_obj.player_dict if other_pid != pid]
                
                # Get trueskill change and new value
                ts_change = trueskill_update_dict[token]["new_trueskill"] - trueskill_update_dict[token]["old_trueskill"]
                new_ts = trueskill_update_dict[token]["new_trueskill"]
                
                message = {
                    "command": "game_over",
                    "outcome": outcomes[pid],
                    "reward": rewards[pid],
                    "trueskill_change": ts_change,
                    "new_trueskill": new_ts,
                    "opponents": opponents,
                    "reason": reason
                }
                
                # Log the game over message with trueskill details
                logging.info(f"Game over for Player {pid}:")
                logging.info(f"  - Outcome: {outcomes[pid]}")
                logging.info(f"  - Reward: {rewards[pid]}")
                logging.info(f"  - New TrueSkill: {new_ts:.2f} (Δ: {ts_change:+.2f})")
                await websocket.send_text(json.dumps(message))
                game_obj.game_over_messages_sent[pid] = True
                logging.info(f"Game over message sent to Player {pid} from update_game_state")
            except Exception as e:
                message_send_exceptions.append((pid, str(e)))
                logging.error(f"Error sending game over message to Player {pid} from update_game_state: {e}")
        else:
            # Mark as sent if websocket doesn't exist
            game_obj.game_over_messages_sent[pid] = True
    
    # Update the overall messages sent flag
    game_obj.all_messages_sent = all(game_obj.game_over_messages_sent.values())
    
    # Log any message sending failures
    if message_send_exceptions:
        logging.error(f"Failed to send game over messages to {len(message_send_exceptions)} players: {message_send_exceptions}")
    else:
        logging.info("Successfully sent game over messages to all connected players")