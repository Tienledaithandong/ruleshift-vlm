import time
import ruleshift.agents.value_selector as vs

ACTIONS = ('PROBE_RED','PROBE_BLUE','PROBE_GREEN','COMMIT_RED','COMMIT_BLUE','COMMIT_GREEN')

def _fixed_payload():
    preds=[]
    for a in ACTIONS:
        color=a.split('_',1)[1].lower()
        under=[]
        for rule in ('red','blue','green'):
            obs='accept' if rule==color else 'reject'
            under.append({'rule_id':rule,'possible_outcomes':[{'observation':obs,'p':1.0}]})
        preds.append({'action_id':a,'under_rules':under,'estimated_future_best_action':f'COMMIT_{color}'})
    return {'candidate_probabilities':{'red':1/3,'blue':1/3,'green':1/3},'action_predictions':preds,'selected_action':'COMMIT_RED'}

belief={'red':1/3,'blue':1/3,'green':1/3}
cost=vs.PublicCostModel(step_cost=-0.05,success_reward=10.0,probe_duration=1,commit_duration=1,lockout_steps=3,remaining_horizon=10,learning_rate=0.3)
payload=_fixed_payload()

# Instrument: wrap the nested V by re-running the choose and adding counters to
# HistoryConsistencyUpdater.update via a subclass that counts calls.
class CountingUpdater(vs.HistoryConsistencyUpdater):
    n_updates = 0
    def update(self, belief, action, outcome):
        self.n_updates += 1
        return super().update(belief, action, outcome)

t0=time.time()
sel=vs.MultiStepValueAwareSelector()
# Build the cost so the selector uses our counting updater.
c=sel.choose(payload,belief,cost)
print('choice:',c,'elapsed %.3fs'%(time.time()-t0))
print('updater updates:', CountingUpdater.n_updates)
