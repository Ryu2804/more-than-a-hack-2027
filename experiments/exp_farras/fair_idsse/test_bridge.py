import unittest
import numpy as np
import bridge as b


class IsolationTests(unittest.TestCase):
    def test_fold_rejects_training_on_test_or_truncation(self):
        heldout=b.MATCHES[0]
        train=b.split_names(heldout)
        b.assert_fold(train,heldout)
        for wrong in (train[:2],(*train[:5],heldout),(*train[:5],train[0])):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):
                b.assert_fold(wrong,heldout)

    def test_predictor_view_never_contains_ground_truth(self):
        value=dict(y=np.array([0]),y25=np.array([0]),X=np.zeros((1,16)),_name='test',teams=np.array([0,1]))
        observations=b.observations(value)
        self.assertNotIn('y',observations)
        self.assertNotIn('y25',observations)
        self.assertEqual(observations['_name'],'test')
        self.assertIn('y',value)

    def test_statistics_use_supplied_training_labels_and_separate_halves(self):
        dataset=dict(y25=np.array([0,0,2,0,0]),blocks25=np.array([[0,2],[2,5]]))
        stats=b.statistics([dataset])
        self.assertAlmostEqual(stats['rate'],2/(5/25/60))
        self.assertAlmostEqual(stats['dead_share'],.2)


if __name__=='__main__':
    unittest.main()
