import itertools
import unittest

import numpy as np

import support as c
import training_models
import segment_decoder as decoder
import adaptive_fusion as fusion
import diagnostics


def prediction(team, dead=None, boundary=None):
    team = np.asarray(team, dtype=float)
    dead = np.zeros(len(team)) if dead is None else np.asarray(dead, dtype=float)
    return dict(probability=np.column_stack(((1-dead)*team, (1-dead)*(1-team), dead)),
        p_dead=dead, p_team=team, boundary=np.full(len(team), .5) if boundary is None else boundary,
        has_boundary=boundary is not None)


class BehaviorTests(unittest.TestCase):
    def test_variable_transition_dp_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(20261113)
        for n in (1, 4, 7):
            home, away = rng.normal(size=(2, n))
            costs = rng.uniform(.1, 2, n)
            path = decoder.viterbi_2state_variable(home, away, costs)
            def score(states):
                return sum(home[t] if states[t] == 0 else away[t] for t in range(n)) - sum(
                    costs[t] for t in range(1, n) if states[t] != states[t-1])
            optimum = max(score(states) for states in itertools.product((0, 1), repeat=n))
            self.assertAlmostEqual(score(path), optimum, places=12)

    def test_dead_breaks_team_memory_even_with_large_transition_cost(self):
        home = np.r_[np.ones(5), np.zeros(10), 0.0]
        away = np.r_[np.zeros(5), np.zeros(10), 0.1]
        dead = np.r_[np.zeros(5, bool), np.ones(10, bool), False]
        labels = decoder._decode_live_team(home, away, np.array([[0, 16]]), dead, np.empty(0), 10.)
        np.testing.assert_array_equal(labels, np.r_[np.zeros(5), np.full(10, 2), 1])

    def test_postmerge_calibration_minimizes_final_count_error(self):
        rng = np.random.default_rng(43)
        team = 1 / (1 + np.exp(-rng.normal(0, 2, 120)))
        observations = dict(X=np.zeros((120,16)), blocks=np.array([[0,120]]))
        spec = next(s for s in decoder.CONFIGS if s['name']=='postmerge')
        output = decoder.decode(prediction(team), observations, spec, dict(rate=15, dead_share=.4))
        base = decoder._previous
        home, away = base._team_evidence(team, observations['X'], observations, spec)
        target = 6
        choices = []
        for penalty in decoder.LAMBDA_GRID:
            labels = base._decode_team(home, away, observations['blocks'], penalty, np.zeros(120,bool))
            base._merge_short_runs(labels, observations['blocks'], 5)
            choices.append((abs(base.n_possessions(labels, observations['blocks'])-target), penalty))
        self.assertEqual((abs(output['actual_count']-target),output['lambda']), min(choices))

    def test_gradient_boundary_resets_at_half_and_dead_gaps(self):
        team=np.array([.01,.99,.01,.99,.01,.99])
        blocks=np.array([[0,3],[3,6]])
        dead=np.array([False,True,False,False,False,False])
        values=decoder._boundary_scores(prediction(team),team,blocks,dead)
        np.testing.assert_array_equal(values[[0,2,3]],0)
        self.assertGreater(values[4],.9)

    def test_unknown_labels_block_auxiliary_boundaries(self):
        labels=np.array([0,1,-1,1,1,1,0])
        targets,valid=training_models.boundary_targets(labels)
        self.assertFalse(valid[0])
        self.assertFalse(valid[2])
        self.assertFalse(valid[3])
        self.assertEqual(targets[2],0)
        self.assertEqual(targets[3],0)
        self.assertEqual(targets[1],1)
        self.assertEqual(targets[6],1)

    def test_short_weights_use_whole_spells_and_known_boundaries(self):
        labels=np.r_[np.zeros(10,int),np.ones(25,int),np.full(10,2),
            np.zeros(10,int),-1,np.ones(12,int),np.zeros(10,int)]
        weights=training_models.short_spell_weights(labels,np.array([[0,45],[45,len(labels)]]))
        np.testing.assert_array_equal(weights[:10],2)
        np.testing.assert_array_equal(weights[10:45],1)
        np.testing.assert_array_equal(weights[45:68],1)
        np.testing.assert_array_equal(weights[68:],2)

    def test_quality_weights_respect_visibility_half_boundaries(self):
        primary=prediction(np.full(24,.5),np.full(24,.5))
        other=prediction(np.ones(24))
        observations=dict(X=np.zeros((24,16)),blocks=np.array([[0,12],[12,24]]))
        observations['X'][:12,3]=1
        output=fusion.combine('quality_old',{},primary,other,observations)
        expected=.6*primary['probability'][0]+.4*other['probability'][0]
        np.testing.assert_allclose(output['probability'][11],expected)
        expected=.95*primary['probability'][0]+.05*other['probability'][0]
        np.testing.assert_allclose(output['probability'][12],expected)
        np.testing.assert_allclose(output['p_team'],output['probability'][:,0]/output['probability'][:,:2].sum(1))

    def test_ensemble_preserves_only_actual_learned_boundary(self):
        learned=prediction(np.full(8,.6),boundary=np.linspace(.1,.8,8))
        neutral=prediction(np.full(8,.4))
        bases=dict(boundary_tcn=learned,short_tcn=neutral,masked_tcn=neutral)
        output=fusion.combine('new_ensemble',bases,neutral,neutral,dict(blocks=np.array([[0,8]])))
        self.assertTrue(output['has_boundary'])
        np.testing.assert_array_equal(output['boundary'],learned['boundary'])

    def test_duration_diagnostics_keep_short_spells(self):
        labels=np.r_[np.zeros(15,int),np.full(10,2),np.ones(30,int),np.full(5,2),np.zeros(80,int)]
        n=len(labels)
        data=dict(y25=np.repeat(labels,5),blocks=np.array([[0,n]]),blocks25=np.array([[0,n*5]]),n_frames=n*5)
        result=diagnostics.summarize(labels,data)
        self.assertEqual(result['n_gt_spells'],3)
        for bucket in result['duration_buckets'].values():
            self.assertEqual(bucket['gt_count'],1)
            for row in bucket['at_iou'].values():
                self.assertEqual(row['matched_recall'],1.)

    def test_diagnostics_split_unknown_and_half_intervals(self):
        labels=np.r_[np.zeros(20,int),np.full(5,-1),np.zeros(20,int)]
        n=len(labels)
        data=dict(y25=np.repeat(labels,5),blocks=np.array([[0,25],[25,n]]),
            blocks25=np.array([[0,125],[125,n*5]]),n_frames=n*5)
        result=diagnostics.summarize(np.zeros(n,int),data)
        self.assertEqual(result['n_gt_spells'],2)
        self.assertEqual(result['n_pred_spells'],2)
        bucket=result['duration_buckets']['4_to_lt_15s']
        self.assertEqual(bucket['gt_count'],2)
        self.assertEqual(bucket['at_iou']['0.9']['matched_count'],2)
        self.assertIsNone(result['duration_buckets']['lt_4s']['at_iou']['0.5']['matched_recall'])


if __name__=='__main__':
    unittest.main(verbosity=2)
