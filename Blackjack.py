import os
import random
import sys
from bot import bet,play

#blackjack foundation provided by: https://gist.github.com/mjhea0/5680216
#betting added by us,

decks = input("Enter number of decks to use: ")
playerOr = int(input("Enter 1 for human player or 0 for bot: "))
runs = int(input("Enter how many runs for the bot: "))



# user chooses number of decks of cards to use
deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*(int(decks)*4)

# initialize scores
wins = 0
losses = 0
ties = 0

#Betting
PlayerMoney = 100
MaxBet = 50000
CurrentBet = 0

def deal(deck):
    hand = []
    for i in range(2):
        if not deck:
            print("HIT")
            deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*(int(decks)*4)
        random.shuffle(deck)
        card = deck.pop()
        if card == 11:card = "J"
        if card == 12:card = "Q"
        if card == 13:card = "K"
        if card == 14:card = "A"
        hand.append(card)
    return hand

def play_again():
    global runs
    print("runs "+ str(runs))
    runs -= 1
    if playerOr == 1:
        again = input("Do you want to play again? (Y/N) : ").lower()
        if again == "y":
            dealer_hand = []
            player_hand = []
            deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*4
            game()
        else:
            print("    \033[1;32;40mWINS:  \033[1;37;40m%s   \033[1;31;40mLOSSES:  \033[1;37;40m%s\n" % (wins, losses))
            print(PlayerMoney)
            print("Bye!")
            sys.exit()
    else:
        if runs > 0:
            dealer_hand = []
            player_hand = []
            deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*4
            game()
        else:
            print("    \033[1;32;40mWINS:  \033[1;37;40m%s   \033[1;31;40mLOSSES:  \033[1;37;40m%s\n" % (wins, losses))
            print(ties)
            print(PlayerMoney)
            print("Bye!")
            sys.exit()

def total(hand):
    total = 0
    for card in hand:
        if card == "J" or card == "Q" or card == "K":
            total+= 10
        elif card == "A":
            if total >= 11: total+= 1
            else: total+= 11
        else: total += card
    return total

def hit(hand):
    global deck
    if not deck:
        print("HIT")
        deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*(int(decks)*4)
    card = deck.pop()
    if card == 11:card = "J"
    if card == 12:card = "Q"
    if card == 13:card = "K"
    if card == 14:card = "A"
    hand.append(card)
    return hand

def clear():
    if os.name == 'nt':
        os.system('CLS')
    if os.name == 'posix':
        os.system('clear')

def print_results(dealer_hand, player_hand):
    #clear()

    # print("\n    WELCOME TO BLACKJACK!\n")
    # print("-"*30+"\n")
    # print("    \033[1;32;40mWINS:  \033[1;37;40m%s   \033[1;31;40mLOSSES:  \033[1;37;40m%s\n" % (wins, losses))
    # print("-"*30+"\n")
    print ("The dealer has a " + str(dealer_hand) + " for a total of " + str(total(dealer_hand)))
    print ("You have a " + str(player_hand) + " for a total of " + str(total(player_hand)))

def blackjack(dealer_hand, player_hand):
    global wins
    global losses
    global ties
    global PlayerMoney
    global CurrentBet

    if total(player_hand) == 21 and total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        print("It's a draw!\n")
        print("You keep your money, you have " + str(PlayerMoney) + "\n")
        ties += 1
        play_again()
    elif total(player_hand) == 21:
        print_results(dealer_hand, player_hand)
        print ("Congratulations! You got a Blackjack!\n")
        PlayerMoney += CurrentBet*2.5
        print("You earnt " + str(CurrentBet*2.5) + ", giving you total of " + str(PlayerMoney) + "\n")
        wins += 1
        play_again()
    elif total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        print ("Sorry, you lose. The dealer got a blackjack.\n")
        PlayerMoney -= CurrentBet
        print("You lost " + str(CurrentBet) + ", giving you total of " + str(PlayerMoney) + "\n")
        losses += 1
        play_again()
    

def score(dealer_hand, player_hand):
    # score function now updates to global win/loss variables
    global wins
    global losses
    global PlayerMoney
    global CurrentBet
    global ties
        
    if total(player_hand) == total(dealer_hand):
        print_results(dealer_hand, player_hand)
        print("It's a draw!\n")
        print("You keep your money, you have " + str(PlayerMoney) + "\n")
        ties += 1
    elif total(player_hand) == 21:
        print_results(dealer_hand, player_hand)
        print ("Congratulations! You got a Blackjack!\n")
        PlayerMoney += CurrentBet*2.5
        print("You earnt " + str(CurrentBet*2.5) + ", giving you total of " + str(PlayerMoney) + "\n")
        wins += 1
    elif total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        print ("Sorry, you lose. The dealer got a blackjack.\n")
        PlayerMoney -= CurrentBet
        print("You lost " + str(CurrentBet) + ", giving you total of " + str(PlayerMoney) + "\n")
        losses += 1
    elif total(player_hand) > 21:
        print_results(dealer_hand, player_hand)
        print ("Sorry. You busted. You lose.\n")
        PlayerMoney -= CurrentBet
        print("You lost " + str(CurrentBet) + ", giving you total of " + str(PlayerMoney) + "\n")
        losses += 1
    elif total(dealer_hand) > 21:
        print_results(dealer_hand, player_hand)
        print ("Dealer busts. You win!\n")
        PlayerMoney += CurrentBet*2
        print("You earnt " + str(CurrentBet*2) + ", giving you total of " + str(PlayerMoney) + "\n")
        wins += 1
    elif total(player_hand) < total(dealer_hand):
        print_results(dealer_hand, player_hand)
        print ("Sorry. Your score isn't higher than the dealer. You lose.\n")
        PlayerMoney -= CurrentBet
        print("You lost " + str(CurrentBet) + ", giving you total of " + str(PlayerMoney) + "\n")
        losses += 1
    elif total(player_hand) > total(dealer_hand):
        print_results(dealer_hand, player_hand)
        print ("Congratulations. Your score is higher than the dealer. You win\n")
        PlayerMoney += CurrentBet*2
        print("You earnt " + str(CurrentBet*2) + ", giving you total of " + str(PlayerMoney) + "\n")
        wins += 1


def game():
    global wins
    global losses
    global PlayerMoney
    global MaxBet
    global CurrentBet

    choice = 0
    #clear()
    print("\n    WELCOME TO BLACKJACK!\n")
    print("-"*30+"\n")
    print("    \033[1;32;40mWINS:  \033[1;37;40m%s   \033[1;31;40mLOSSES:  \033[1;37;40m%s\n" % (wins, losses))
    print("-"*30+"\n")
    if playerOr == 1:
        CurrentBet = int(input("How much do you want to bet? Maximum of 50000\n"))
        while CurrentBet > MaxBet or CurrentBet <= 0:
            print("You can't bet that amount")
            CurrentBet = int(input("How much do you want to bet? Maximum of 50000\n"))
    else:
        CurrentBet = bet()
        print(CurrentBet)

    dealer_hand = deal(deck)
    player_hand = deal(deck)
    print ("The dealer is showing a " + str(dealer_hand[0]))
    print ("You have a " + str(player_hand) + " for a total of " + str(total(player_hand)))
    blackjack(dealer_hand, player_hand)
    quit=False
    while not quit:
        if playerOr == 1:
            choice = input("Do you want to [H]it, [S]tand, or [Q]uit: ").lower()
        else:
            choice = play()
        if choice == 'h':
            hit(player_hand)
            print(player_hand)
            print("Hand total: " + str(total(player_hand)))
            if total(player_hand)>21:
                score(dealer_hand,player_hand)
                play_again()
        elif choice=='s':
            while total(dealer_hand)<17:
                hit(dealer_hand)
                print(dealer_hand)
            score(dealer_hand,player_hand)
            play_again()
        elif choice == "q":
            print("Bye!")
            quit=True
            sys.exit()


if __name__ == "__main__":
   game()